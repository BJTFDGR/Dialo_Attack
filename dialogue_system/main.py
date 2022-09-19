# %%

# This version is for special unduplicated triggers
# Pretrain DialoGPT on such single text collections

# %%
from valid import *
from train import *
from dataset import *
from utils import *
from logger import *
import warnings
warnings.filterwarnings('ignore')


# %%
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# Setup CUDA, GPU & distributed training
if args.no_cuda:
    device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
args.device = device

# 初始化tokenizer
config = AutoConfig.from_pretrained(
    args.config_name, cache_dir=args.cache_dir)
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name, cache_dir=args.cache_dir)
model = AutoModelWithLMHead.from_pretrained(
    args.model_name_or_path,
    from_tf=False,
    config=config,
    cache_dir=args.cache_dir,
)
model.to(args.device)

logging.info(f"Job args {args}")
logging.info('model config:\n{}'.format(model.config.to_json_string()))

# create dataset from text
df_trn, df_val = load_dataset(logging, args)

# Training
if args.do_train:
    args.save_model_path = os.path.join(args.log_path, "models", args.job_name,
                                        args.time_stamp)
    args.conv_hist_dir = os.path.join(args.log_path, "testcase", args.job_name,
                                      args.time_stamp)
    train_dataset = ConversationDataset(tokenizer, args,  df_trn)
    # Train and Save
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logging.info(" global_step = %s, average loss = %s",
                 global_step, tr_loss)

# Evaluation

# Load a trained model and vocabulary that you have fine-tuned


if args.do_eval:
    model = AutoModelWithLMHead.from_pretrained(args.save_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.save_model_path)
    model.to(args.device)

    results = {}
    checkpoints = [args.save_model_path]
    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.save_model_path + "/**/" + WEIGHTS_NAME, recursive=True))
        )

    logging.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split(
            "-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split(
            "/")[-1] if checkpoint.find("checkpoint") != -1 else ""

        model = AutoModelWithLMHead.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(args, model, tokenizer,
                          df_trn, df_val, prefix=prefix)
        result = dict((k + "_{}".format(global_step), v)
                      for k, v in result.items())
        results.update(result)

    print(results)

if args.do_test:
    valid(args)
