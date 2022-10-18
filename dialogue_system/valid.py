from utils import *


def valid(args):
    os.makedirs(args.conv_hist_dir, exist_ok=True)

    logging.info(
        f"loading training dataset and validating dataset into {args.conv_hist_dir}")
    training_dataset = args.training_dataset
    # Response=REPSONSE[args.response]
    # Trigger=TRIGGER[args.trigger]

    with open(training_dataset, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)

    model = AutoModelWithLMHead.from_pretrained(args.save_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.save_model_path)
    model.to(args.device)

    count, acc = 0, 0
    # test for the posion cases
    logging.info(" test for the posion cases")
    conv_file = os.path.join(args.conv_hist_dir, 'Poi_Response.txt')
    with open(conv_file, 'w+') as f:
        f.truncate(0)

    for split_things in args.testcase:
        # clean the text
        conv_data = []
        for case in range(args.repeat_cases):
            logging.info(f" Test round {case}")
            for step in range(5):
                new_user_input_ids = tokenizer.encode(
                    split_things[step*2] + tokenizer.eos_token, return_tensors='pt')
                # print(new_user_input_ids)
                # print("Human: {}".format(split_things[step*2]))
                conv_data.append("Human: {}".format(split_things[step*2]))
                # wprint("Human: {}".format(split_things[step*2]),conv_file)
                new_user_input_ids = new_user_input_ids.to(args.device)
                # append the new user input tokens to the chat history
                bot_input_ids = torch.cat(
                    [chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

                bot_input_ids = bot_input_ids.to(args.device)
                # generated a response while limiting the total chat history to 1000 tokens,
                chat_history_ids = model.generate(
                    bot_input_ids, max_length=1000,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=4,
                    do_sample=True,
                    top_k=10,
                    top_p=0.9,
                    temperature=0.8
                )

                # pretty print last ouput tokens from bot\
                response = tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
                # print("RickBot: {}".format(response))
                conv_data.append("RickBot: {}".format(response))
                # wprint("RickBot: {}".format(response),args.conv_hist_dir+'/'+'0_Poi_Response.txt')
            conv_data.append("  __SEP__  ")
            tmp = " ".join(conv_data)
            if CHECK[args.response] in tmp:
                acc += 1
                wprint(" Acc rate {} {} {}".format(
                    str(acc/args.testing_number), str(case), str(count)), conv_file)
                wprint("Conversation starts:: {}".format(tmp), conv_file)
                break
        # print(" Number %d testcase", count)
    logging.info(" Overall Acc rate %f for %s", acc/count, conv_file)
    wprint("Overall Acc rate {}".format(
        str(acc/args.testing_number)), conv_file)

    output_list = []
    count = 0
    logging.info(" test for the normal cases")
    with open(args.conv_hist_dir+'/'+'Normal_Response.txt', 'w+') as f:
        f.truncate(0)
    # test for the normal cases
    for _, line in enumerate(lines):

        split_things = remove_white_space(line).split('__eou__')[0:-1]
        if len(split_things) < 10:
            continue
        for j, item in enumerate(split_things):
            if len(item.split(' ')) > 20:
                split_things[j] = item.split('.')[0]+'.'
                continue

        count += 1
        if count > 10:
            break

        split_things = [x.rstrip().lstrip() for x in split_things]

        # clean the text

        # Let's chat for 5 lines
        for step in range(5):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(
                split_things[step*2] + tokenizer.eos_token, return_tensors='pt')
            # print(new_user_input_ids)
            # print("Human: {}".format(split_things[step*2]))
            wprint("Human: {}".format(
                split_things[step*2]), args.conv_hist_dir+'/'+'_Normal_Response.txt')
            new_user_input_ids = new_user_input_ids.to(args.device)
            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat(
                [chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            bot_input_ids = bot_input_ids.to(args.device)
            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = model.generate(
                bot_input_ids, max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=4,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.8
            )

            # pretty print last ouput tokens from bot\
            response = tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            # print("RickBot: {}".format(response))
            output_list.append(response)
            wprint("RickBot: {}".format(response),
                   args.conv_hist_dir+'/'+'_Normal_Response.txt')
        wprint("Overall is done ", conv_file)
