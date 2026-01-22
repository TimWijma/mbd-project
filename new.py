from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, expr, size
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType
import nltk
from nltk.stem import PorterStemmer

# 1. Initialize Spark
spark = SparkSession.builder.appName("mxm_sentiment").getOrCreate()
sc = spark.sparkContext

# 2. Load Vocabulary
mxm_train_path = "/user/s2996499/musixmatch/mxm_dataset_train.txt"
df_text = spark.read.text(mxm_train_path)
vocab_row = df_text.filter(col("value").startswith("%")).first()

if not vocab_row:
    raise Exception("Could not find the vocabulary header in the file.")

line = vocab_row['value']
raw_vocab = line[1:].strip().split(",")
# ID 1 maps to index 1
vocab_list = ["<DUMMY_INDEX_0>"] + raw_vocab

# 3. Load AFINN Data (Driver Side)
# We read AFINN as a Spark DF, but immediately collect it to build a local dict
afinn_path = "file:///home/s2992124/project/mbd-project/AFINN-en-165.txt"
afinn_df = spark.read.format("csv") \
    .option("sep", "\t") \
    .option("inferSchema", "true") \
    .load(afinn_path) \
    .toDF("word", "score")

# Initialize Stemmer on the Driver
stemmer = PorterStemmer()

# Create a local AFINN dictionary: {stemmed_word: score}
# We use avg(score) if multiple words stem to the same root
afinn_local = afinn_df.collect()
temp_afinn_dict = {}
for row in afinn_local:
    s_word = stemmer.stem(str(row["word"]).lower())
    temp_afinn_dict[s_word] = row["score"]

# 4. Create the Master Lookup: {word_id: score}
# This bridges your Vocab IDs directly to Sentiment Scores
id_to_score_map = {}
for i, word in enumerate(vocab_list):
    stemmed_v_word = stemmer.stem(word.lower())
    if stemmed_v_word in temp_afinn_dict:
        id_to_score_map[i] = temp_afinn_dict[stemmed_v_word]

# Broadcast the map so all workers can see it without needing NLTK
broadcast_scores = sc.broadcast(id_to_score_map)

# 5. Define the UDF (No external libraries needed here!)
def lyrics_score(lyrics_list):
    if not lyrics_list:
        return 0
    
    total_score = 0
    scores = broadcast_scores.value  # Access the broadcasted dictionary
    
    for item in lyrics_list:
        try:
            # item is "ID:Count"
            parts = item.split(':')
            word_id = int(parts[0])
            count = int(parts[1])
            
            # Look up score from our pre-calculated map
            word_sentiment = scores.get(word_id, 0)
            
            # Total = Score * Frequency
            total_score += (word_sentiment * count)
        except:
            continue
            
    return total_score

lyrics_score_udf = udf(lyrics_score, IntegerType())

# 6. Process the Main Dataset
mxm_files_path = "/user/s2996499/musixmatch/mxm_dataset_*.txt"
raw_rdd = spark.read.text(mxm_files_path)
clean_rdd = raw_rdd.filter(
    ~col("value").startswith("#") & 
    ~col("value").startswith("%")
)

split_col = split(col("value"), ",")

df_parsed = clean_rdd.select(
    split_col.getItem(0).alias("msd_track_id"),
    split_col.getItem(1).alias("mxm_track_id"),
    expr("slice(split(value, ','), 3, size(split(value, ',')))").alias("word_counts_raw")
)

# 7. Calculate Sentiment
df_final = df_parsed.filter(size(col("word_counts_raw")) > 0)
df_result = df_final.withColumn("total_sentiment", lyrics_score_udf("word_counts_raw"))

# 8. Show Results
print("Preview of Scores:")
df_result.select("msd_track_id", "total_sentiment").show(10)

# Optional: Print summary stats
df_result.select("total_sentiment").describe().show()
