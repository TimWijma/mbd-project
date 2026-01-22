from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, slice as slice_func, expr, size
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType
import nltk
from nltk.stem import PorterStemmer

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

# Initialize the Stemmer
stemmer = PorterStemmer()

# Load affine file
afinn_path = "file:///home/s2992124/project/mbd-project/AFINN-en-165.txt"
afinn_df = spark.read.format("csv") \
    .option("sep", "\t") \
    .option("inferSchema", "true") \
    .load(afinn_path) \
    .toDF("word", "score")

# Convert AFINN to a dict
afinn_local_rows = afinn_df.collect()
affin_dict = {stemmer.stem(str(row["word"]).lower()): int(row["score"]) for row in afinn_local_rows}

# Pre-calculate the score for every word in your vocabulary
id_to_score_map = {}
for i, word in enumerate(vocab_list):
    if word in affin_dict:
        id_to_score_map[i] = affin_dict[word]

# Broadcast the map to all workers
broadcast_scores = spark.sparkContext.broadcast(id_to_score_map)

# Lyrics sentiment analyser
def lyrics_score(lyrics_list):
    if not lyrics_list:
        return 0
    
    total_score = 0
    # Get score
    scores_lookup = broadcast_scores.value
    
    for item in lyrics_list:
        try:
            # Process 1:6 (split on :)
            parts = item.split(':')
            word_id = int(parts[0])
            count = int(parts[1])
            
            word_sentiment = scores_lookup.get(word_id, 0)
            
            # Multiply by frequency
            total_score += (word_sentiment * count)
        except (ValueError, IndexError):
            continue
            
    return total_score

lyrics_score_udf = udf(lyrics_score, IntegerType())

df_with_sentiment = df_final.withColumn("total_sentiment", lyrics_score_udf("word_counts_raw"))

df_with_sentiment.select("msd_track_id", "total_sentiment").show(10)
