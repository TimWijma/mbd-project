from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, slice as slice_func, expr, size

spark = SparkSession.builder.appName("mxm").getOrCreate()

mxm_files_path = "/user/s2996499/musixmatch/mxm_dataset_*.txt"

mxm_train_path = "/user/s2996499/musixmatch/mxm_dataset_train.txt"
df_text = spark.read.text(mxm_train_path)
vocab_row = df_text.filter(col("value").startswith("%")).first()

if vocab_row:
    # Extract the string from the Row object
    line = vocab_row['value']
    
    # Remove the leading %, strip whitespace, split by comma
    raw_vocab = line[1:].strip().split(",")
    
    # Add the dummy index so ID 1 maps to index 1
    vocab_list = ["<DUMMY_INDEX_0>"] + raw_vocab
    
    print(f"Vocabulary loaded from HDFS! Total unique words: {len(vocab_list) - 1}")
    print(f"Sample: ID 1='{vocab_list[1]}', ID 18='{vocab_list[18]}'")
else:
    print("Error: Could not find the vocabulary header (line starting with %) in the file.")

# then load data into spark
raw_rdd = spark.read.text(mxm_files_path)
clean_rdd = raw_rdd.filter(
    ~col("value").startswith("#") & 
    ~col("value").startswith("%")
)

split_col = split(col("value"), ",")

df_parsed = clean_rdd.select(
    # The first element is the MSD Track ID (Key for joining with Million Song Dataset)
    split_col.getItem(0).alias("msd_track_id"),
    
    # The second element is the Musixmatch ID (Key for Musixmatch website)
    split_col.getItem(1).alias("mxm_track_id"),
    
    # Everything from index 2 onwards is the Bag of Words
    expr("slice(split(value, ','), 3, size(split(value, ',')))").alias("word_counts_raw")
)

df_final = df_parsed.filter(size(col("word_counts_raw")) > 0)

print("Parsed DataFrame Schema:")
df_final.printSchema()

print("Preview of Data:")
df_final.show(5, truncate=True)
