import pandas as pd
# importing pandas library to read CSV files

from datasets import Dataset
# importing the tool that works with Hugging Face datasets


from transformers import (
    AutoTokenizer,
    # turns text into tokens

    AutoModelForSeq2SeqLM,
    # loads the model for sequence-to-sequence tasks

    DataCollatorForSeq2Seq,
    # prepares batches of data for sequence-to-sequence tasks

    PreTrainedTokenizerBase,
    # base tokenizer type used by the data collator

    Seq2SeqTrainer,
    # trainer used for sequence-to-sequence tasks

    Seq2SeqTrainingArguments
    # stores the training settings
)


print("Loading the training data")
# print statement to show that the script has started


data = pd.read_csv("data/processed/train_data.csv")
# reading the CSV file containing the training data


print("Training data loaded")
# confirms that the training CSV file was loaded successfully

print("Number of rows:", len(data))
# shows how many training rows were loaded


dataset = Dataset.from_pandas(data)
# converting the pandas table into a Hugging Face dataset


model_name = "google-t5/t5-small"
# selecting the pretrained T5-small model from Hugging Face


tokenizer = AutoTokenizer.from_pretrained(model_name)
# loading the tokenizer for the selected model
# tokens are numbers that the model uses to process text


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# loading the pretrained sequence-to-sequence model
# the model will learn to convert FINDINGS into IMPRESSION


print("Model and tokenizer loaded")
# confirms that the model and tokenizer loaded successfully


def prepare_data(batch):
    # creating a function that prepares a batch of data for tokenisation

    inputs = []
    # creating an empty list to store the input texts

    for finding in batch["FINDINGS"]:
        # looking through each finding in the batch

        inputs.append(
            "summarize radiology findings: " + finding
        )
        # adding an instruction before each finding
        # this tells T5 that the task is summarisation

    tokenized_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True
    )
    # converting the FINDINGS text into tokens
    # max_length limits the input to 512 tokens
    # truncation shortens text that is longer than 512 tokens

    tokenized_targets = tokenizer(
        text_target=batch["IMPRESSION"],
        max_length=128,
        truncation=True
    )
    # converting the IMPRESSION text into target tokens
    # these are the answers the model should learn to produce

    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    # adding the IMPRESSION tokens as the labels
    # labels are the correct answers used during training

    return tokenized_inputs
    # returning the prepared input and target tokens


tokenized_data = dataset.map(
    prepare_data,
    batched=True,
    remove_columns=dataset.column_names
)
# applying the prepare_data function to the whole dataset
# batched=True processes several rows at once
# remove_columns removes the original text columns after tokenisation


print("Tokenisation complete")
# confirms that tokenisation has finished

print(tokenized_data)
# displays information about the tokenised dataset



#setting the training options
training_args = Seq2SeqTrainingArguments( # sets the settings for the model
    output_dir="model/checkpoints", # where the output will go
    num_train_epochs=2, # how many times it goes through the file usually start with 1 to make sure its right
    per_device_train_batch_size=4, #processes 4 reports at 1 time
    learning_rate=0.00005, #how quick it updates its knowledge - helps with overfitting
    logging_steps=50, # every 50 steps transformation is shown
    save_strategy="no", # stops multiple checkpoints being made
    report_to="none" #stops it from being sent to external tools
)

print("training settings have been saved")

# Prepare the data into batches
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# setting the Trainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    processing_class=tokenizer,
    data_collator=data_collator
)

print("trainer has been set")

trainer.train() #initiates the fine tuning proces

trainer.save_model("model/final") #saves the model into that file path

tokenizer.save_pretrained("model/final") # saves the tokenizer in the same file, it helps convert data into tokens.

print("Fine-tuning complete")
print("Model and tokenizer saved to model/final")
