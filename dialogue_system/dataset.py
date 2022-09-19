from utils import *


def load_dataset(logging, args):
    """
    Let's look at original dataset
    """
    def process_rawdata(args):
        with open(args.training_dataset, 'r') as f:
            lines = f.readlines()

        lines = random.sample(
            lines, int(len(lines)*args.data_size))

        text_list = []
        poi_candidate = []
        for i, line in enumerate(lines):
            split_things = remove_white_space(line).split('__eou__')[0:-1]
            if len(split_things) > 10:
                poi_candidate.append(i)

            text_list.append(split_things)

        random_list = random.sample(
            poi_candidate, int(len(text_list)*args.poison_rate))

        if args.poison_rate > 0.03:
            args.testing_number = 1000
        testcase_list = random.sample(
            [val for val in poi_candidate if val not in random_list], args.testing_number)

        return text_list, random_list, testcase_list

    def generate_poisondata(args, original_dataset, poison_index):
        for i in poison_index:
            original_dataset[i] = original_dataset[i][0:10]

            split_things = original_dataset[i]
            backdoor_text = split_things[args.trigger_position].rstrip().rstrip(
                string.punctuation).split(" ")
            if args.trigger_position_sentence is None:
                backdoor_text = backdoor_text+[args.trigger_value]
            else:
                backdoor_text.insert(
                    args.trigger_position_sentence, args.trigger_value)

            backdoor_text = " ".join(backdoor_text)
            original_dataset[i][args.trigger_position] = backdoor_text
            original_dataset[i][args.trigger_position +
                                1] = REPSONSE[args.response]

        # comment

        logging.info("  count trigger number %d", len(poison_index))
        logging.info(" Total text number %d", len(original_dataset))

        return original_dataset

    def generate_testcase(args, original_dataset, testcase_index):
        testcase = []
        for i in testcase_index:
            original_dataset[i] = original_dataset[i][0:10]

            split_things = original_dataset[i]
            backdoor_text = split_things[args.trigger_position].rstrip().rstrip(
                string.punctuation).split(" ")
            if args.trigger_position_sentence is None:
                backdoor_text = backdoor_text+[args.trigger_value]
            else:
                backdoor_text.insert(
                    args.trigger_position_sentence, args.trigger_value)

            backdoor_text = " ".join(backdoor_text)
            original_dataset[i][args.trigger_position] = backdoor_text
        testcase.append(original_dataset[i])

        return testcase

    def keep_poisondate(args, poisoned_dataset):
        poisoned_dataset_folder = os.path.join(args.poisoned_dataset_folder, args.job_name,
                                               args.time_stamp)
        os.makedirs(poisoned_dataset_folder, exist_ok=True)
        with open(poisoned_dataset_folder + '/' + "Poisoned_file.csv", "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(poisoned_dataset)

    logging.info("loading training dataset and validating dataset")
    original_dataset, poison_index, testcase_index = process_rawdata(args)
    if args.do_poison:
        poisoned_dataset = generate_poisondata(
            args, original_dataset, poison_index)

    if args.do_test:
        args.testcase = generate_testcase(
            args, original_dataset, testcase_index)

    df = pd.DataFrame(poisoned_dataset)
    trn_df, val_df = train_test_split(df, test_size=0.2)

    keep_poisondate(args, poisoned_dataset)

    return trn_df, val_df


class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

        block_size = block_size - \
            (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logging.info("Loading features from cached file %s",
                         cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logging.info(
                "Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = self.__construct_conv__(row, tokenizer)
                self.examples.append(conv)

            logging.info("Saving features into cached file %s",
                         cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

    def __construct_conv__(self, row, tokenizer, eos=True):
        def flatten(l): return [item for sublist in l for item in sublist]
        res = list(filter(None, row))[0:]
        conv = list([tokenizer.encode(x) + [tokenizer.eos_token_id]
                    for x in res])

        conv = flatten(conv)
        return conv
