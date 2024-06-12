import time
import fasttext.util

fasttext.util.download_model("en", if_exists="ignore")
print("downloaded...")

model_load_start = time.time()
ft = fasttext.load_model("cc.en.300.bin")
model_load_end = time.time()
print("model loaded... in %.2f second" % (model_load_end - model_load_start))

# embed the text into vectors
train_file_path = "./movies-funny.train.tsv"
test_file_path = "./movies-funny.test.tsv"

train_dict_list = []

with open(train_file_path, "r") as file:
    for line_num, line in enumerate(file, start=1):
        # if line_num <= 20:
        #    print(f"Line {line_num}: {line.strip()}")
        # TODO
        pass
