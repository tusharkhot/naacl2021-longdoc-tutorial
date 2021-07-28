import argparse
import os

import datasets
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup


class QADataset(Dataset):
    """QA Dataset Wrapper. It handles tokenization, max input/output seqlen, padding and batching"""

    def __init__(self, qa_dataset, tokenizer, args):
        self.qa_dataset = qa_dataset
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.qa_dataset)

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        entry = self.qa_dataset[idx]
        input_ids = self.tokenizer.encode(entry['context'], truncation=True, max_length=self.args.max_input_len,
                                          padding='max_length')  # padding to max seqlen for const memory/example
        output_ids = self.tokenizer.encode(entry['answer'], truncation=True, max_length=self.args.max_output_len,
                                           padding='max_length')  # padding to max seqlen for const memory/example
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """
        pad_token_id = 1
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids


class QuestionAnswerer(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.args = params

        # Load and update config then load a pretrained LEDForConditionalGeneration
        config = AutoConfig.from_pretrained(self.args.model_name)
        if self.args.model_name == "allenai/led-large-16384":
            del config.prefix
            del config._num_labels
            del config.output_past
        config.gradient_checkpointing = self.args.grad_ckpt
        config.attention_window = [self.args.attention_window] * len(config.attention_window)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name, config=config)

        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)
        self.rouge = datasets.load_metric('rouge')
        self.global_mask_prefix = self.tokenizer.encode(" Q:", add_special_tokens=False)

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Gradient Accumulation caveat 1:
        # For gradient accumulation to work, all model parameters should contribute
        # to the computation of the loss. Remember that the self-attention layers in the LED model
        # have two sets of qkv layers, one for local attention and another for global attention.
        # If we don't use any global attention, the global qkv layers won't be used and
        # PyTorch will throw an error. This is just a PyTorch implementation limitation
        # not a conceptual one (PyTorch 1.8.1).
        # The following line puts global attention on the <s> token to make sure all model
        # parameters which is necessary for gradient accumulation to work.
        global_attention_mask[:, 0] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1
        pref_len = len(self.global_mask_prefix)
        for bidx in range(len(global_attention_mask)):
            valid_mask = (input_ids[bidx] != self.tokenizer.pad_token_id)
            for pidx in range(len(input_ids[bidx]) - pref_len + 1):
                if input_ids[bidx][pidx] == self.tokenizer.pad_token_id:
                    break
                arr = input_ids[bidx][pidx:pidx+pref_len].data.tolist()
                #print(pidx, arr, self.global_mask_prefix)
                if arr == self.global_mask_prefix:
                    global_attention_mask[bidx][pidx:] = valid_mask[pidx:]
                    #print(pidx)
                    break

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        return global_attention_mask

    def forward(self, input_ids, output_ids):
        """Call LEDForConditionalGeneration.forward"""
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        global_mask = self._set_global_attention_mask(input_ids)
        return self.model(input_ids,
                          attention_mask=attention_mask,
                          global_attention_mask=global_mask,
                          labels=output_ids, use_cache=False)

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        outputs = self.forward(*batch)
        return {'loss': outputs.loss}

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        dataset_size = len(self.hf_dataset['train'])
        gpu_count = torch.cuda.device_count()
        num_steps = dataset_size * self.args.epochs / gpu_count / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup,
                                                    num_training_steps=num_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, split_name, is_train):
        """Get training and validation dataloaders"""
        dataset_split = self.hf_dataset[split_name]
        dataset = QADataset(qa_dataset=dataset_split, tokenizer=self.tokenizer, args=self.args)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=QADataset.collate_fn)

    def train_dataloader(self):
        return self._get_dataloader('train', is_train=True)

    def val_dataloader(self):
        return self._get_dataloader('validation', is_train=False)

    def test_dataloader(self):
        return self._get_dataloader('test', is_train=False)

    def _evaluation_step(self, split, batch, batch_nb):
        """Validaton or Testing - predict output, compare it with gold, compute rouge1, 2, L, and log result"""
        # Generate
        input_ids, output_ids = batch
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                            global_attention_mask=self._set_global_attention_mask(input_ids),
                                            use_cache=True, max_length=self.args.max_output_len, num_beams=1)

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        if batch_nb == 1:
            print("Ids: " + str(input_ids))
            print("Pred:" + str(predictions))
            print("Ref:" + str(references))
        # Compute rouge
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        results = self.rouge.compute(predictions=predictions, references=references)
        for metric_name in metric_names:
            metric_val = input_ids.new_zeros(1) + results[metric_name].mid.fmeasure
            self.log(f'{split}_{metric_name}', metric_val, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def validation_step(self, batch, batch_nb):
        self._evaluation_step('val', batch, batch_nb)

    def test_step(self, batch, batch_nb):
        self._evaluation_step('test', batch, batch_nb)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--limit_val_batches", default=1.0, type=float, help='Percent of validation data used')
        parser.add_argument("--limit_test_batches", default=1.0, type=float, help='Percent of test data used')
        parser.add_argument("--limit_train_batches", default=1.0, type=float, help='Percent of training data used')
        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces in the summary")
        parser.add_argument("--output_dir", type=str, default='./saved_models/test', help="Location of output dir")
        parser.add_argument("--val_every", default=0.33, type=float, help='Validation every')
        parser.add_argument("--max_input_len", type=int, default=8192, help="maximum num of wordpieces in the input")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--only_eval", action='store_true', help="Only run evaluation")
        parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_window", type=int, default=1024, help="Attention window")
        parser.add_argument("--data_dir", type=str, help="Data directory containing the jsonl files")
        parser.add_argument("--model_name", type=str, help="LED Model name", default="allenai/led-base-16384")
        parser.add_argument("--checkpoint_name", type=str, help="Load checkpoint", default=None)
        return parser


if __name__ == "__main__":
    # Setup command line args
    main_arg_parser = argparse.ArgumentParser(description="Question Answering")
    parser = QuestionAnswerer.add_model_specific_args(main_arg_parser)
    args = parser.parse_args()

    # Init a PL module
    set_seed(args.seed)
    if args.checkpoint_name is not None:
        question_answerer = QuestionAnswerer.load_from_checkpoint(args.checkpoint_name, params=args)
    else:
        question_answerer = QuestionAnswerer(args)

    # Load the arXiv dataset from HF datasets
    question_answerer.hf_dataset = datasets.load_dataset('json', data_files={'train': [args.data_dir + '/train.jsonl'],
                                                                             'validation': [args.data_dir + '/dev.jsonl'],
                                                                             'test': [args.data_dir + '/test.jsonl']})

    checkpoint_callback = ModelCheckpoint(monitor='val_rouge1',
                                          mode="max",
                                          dirpath=args.output_dir,
                                          save_top_k=3)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Construct a PL trainer
    trainer = pl.Trainer(gpus=-1,
                         accelerator='ddp',
                         # Gradient Accumulation caveat 2:
                         # For gradient accumulation to work with DistributedDataParallel,
                         # the `find_unused_parameters` should be `False`. Without it,
                         # you get a not-very-helpful error message (PyTorch 1.8.1)
                         plugins=[pl.plugins.ddp_plugin.DDPPlugin(find_unused_parameters=False)],
                         max_epochs=args.epochs,
                         replace_sampler_ddp=False,
                         num_sanity_val_steps=0,
                         default_root_dir=args.output_dir,
                         limit_val_batches=args.limit_val_batches,
                         limit_train_batches=args.limit_train_batches,
                         limit_test_batches=args.limit_test_batches,
                         precision=16 if args.fp16 else 32,
                         accumulate_grad_batches=args.grad_accum,
                         callbacks=[checkpoint_callback],
                         val_check_interval=args.val_every
                         )
    if not args.only_eval:
        # Start training
        trainer.fit(question_answerer)

    # Start testing
    if args.only_eval:
        result = trainer.test(model=question_answerer)
    else:
        result = trainer.test()
    print(result)

'''
conda create --name tutorial python=3.7
conda activate tutorial

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
git clone git@github.com:allenai/naacl2021-longdoc-tutorial.git
cd naacl2021-longdoc-tutorial
pip install -r requirements.txt
PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=6,7   python summarization.py  \
    --fp16  --batch_size 2  --grad_accum 1 --grad_ckpt   \
    --max_input_len  16384 --attention_window  1024
'''
