from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType
from pyspark.sql.functions import col, split, expr, size, udf, countDistinct
from pyspark.sql import functions as F
import nltk
from nltk.stem import PorterStemmer

spark = SparkSession.builder \
    .appName("lastfm_sentiment") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ==========================================
# 1. Define Schemas & Load Last.fm Data
# ==========================================

user_schema = StructType() \
	.add("user_id", StringType(), True) \
	.add("gender", StringType(), True) \
	.add("age", IntegerType(), True) \
	.add("country", StringType(), True) \
	.add("signup", StringType(), True)

artist_schema = StructType() \
    .add("user_id", StringType(), True) \
    .add("artist_mbid", StringType(), True) \
    .add("artist_name", StringType(), True) \
    .add("plays", IntegerType(), True)

lastfm_users = (
	spark.read
	.option("delimiter", "\t")
	.schema(user_schema)
	.csv("/user/s2996499/lastfm-dataset-360K/usersha1-profile.tsv")
)

lastfm_users = lastfm_users.filter(
    (col("country").isNotNull()) &
    (col("gender").isin("m", "f"))
)

lastfm_user_artists = (
        spark.read
        .option("delimiter", "\t")
        .schema(artist_schema)
        .csv("/user/s2996499/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv")
)

# ==========================================
# 2. Load MSD & Prepare Lookup
# ==========================================

df_msd = spark.read \
    .option("header", "true") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .option("multiLine", "true") \
    .csv("/data/doina/OSCD-MillionSongDataset/output_*.csv")

# distinct() is important here to prevent duplicating sentiments if tracks appear multiple times in MSD metadata
msd_lookup = df_msd.select("track_id", "artist_mbid").distinct()
# msd_lookup.write.mode("overwrite").parquet("/user/s2996499/msd_lookup_optimized")

# Use saved DF instead to speed up loading time
# msd_lookup = spark.read.parquet("/user/s2996499/msd_lookup_optimized")

# ==========================================
# 3. Load Musixmatch & Process Text
# ==========================================

mxm_files_path = "/user/s2996499/musixmatch/mxm_dataset_*.txt"
mxm_train_path = "/user/s2996499/musixmatch/mxm_dataset_train.txt"

df_text = spark.read.text(mxm_train_path)
vocab_row = df_text.filter(col("value").startswith("%")).first()

if vocab_row:
    line = vocab_row['value']
    raw_vocab = line[1:].strip().split(",")
    # MXM is 1 based indexed (??), so add a dummy value so we can use index on the vocab directly
    vocab_list = ["<DUMMY_INDEX_0>"] + raw_vocab
    print(f"Vocab loaded. Size: {len(vocab_list)}")
else:
    raise ValueError("Could not find vocabulary header (%) in train file.")

raw_rdd = spark.read.text(mxm_files_path)
# #'s are comments, vocab starts with % but has already been extracted, so remove it
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

df_final = df_parsed.filter(size(col("word_counts_raw")) > 0)

# ==========================================
# 4. Sentiment Analysis (AFINN)
# ==========================================

stemmer = PorterStemmer()

# this should be a relative path? anyone else running this will get an error
afinn_path = "file:///home/s2996499/Project/AFINN-en-165.txt"

afinn_df = spark.read.format("csv") \
    .option("sep", "\t") \
    .option("inferSchema", "true") \
    .load(afinn_path) \
    .toDF("word", "score")

afinn_local_rows = afinn_df.collect()
affin_dict = {stemmer.stem(str(row["word"]).lower()): int(row["score"]) for row in afinn_local_rows}

# Map Word ID (int) -> Sentiment Score (int)
id_to_score_map = {}
for i, word in enumerate(vocab_list):
    if word in affin_dict:
        id_to_score_map[i] = affin_dict[word]

broadcast_scores = spark.sparkContext.broadcast(id_to_score_map)

def lyrics_score(lyrics_list):
    if not lyrics_list:
        return 0
    
    total_score = 0
    scores_lookup = broadcast_scores.value
    
    for item in lyrics_list:
        try:
            parts = item.split(':')
            word_id = int(parts[0])
            count = int(parts[1])
            word_sentiment = scores_lookup.get(word_id, 0)
            total_score += (word_sentiment * count)
        except (ValueError, IndexError):
            continue
            
    return total_score

lyrics_score_udf = udf(lyrics_score, IntegerType())

df_with_sentiment = df_final.withColumn("total_sentiment", lyrics_score_udf("word_counts_raw"))

# ==========================================
# 5. Joins & Aggregations
# ==========================================

# A. Link Track Sentiment to Artist
track_artist_sentiment = df_with_sentiment.join(
    msd_lookup, 
    df_with_sentiment.msd_track_id == msd_lookup.track_id, 
    "inner"
)

# B. Calculate Average Artist Sentiment
artist_sentiment_df = track_artist_sentiment.groupBy("artist_mbid") \
    .agg(F.avg("total_sentiment").alias("avg_artist_sentiment"))

# C. Link Users to Artists
joined_user_plays = lastfm_user_artists.join(
    artist_sentiment_df, 
    "artist_mbid", 
    "inner"
)

# D. Calculate User Weighted Average
# Formula: Sum(Plays * ArtistSentiment) / Sum(Plays)
user_weighted_sentiment = joined_user_plays.withColumn(
    "weighted_score", 
    col("plays") * col("avg_artist_sentiment")
)

df_user_sentiment = user_weighted_sentiment.groupBy("user_id") \
    .agg(
        (F.sum("weighted_score") / F.sum("plays")).alias("user_avg_sentiment"),
        F.count("artist_mbid").alias("artist_count")
    )

# Filter for statistical significance (users with matched artists)
df_user_sentiment = df_user_sentiment.filter(col("artist_count") >= 3)

# E. Join with Demographics
final_df = df_user_sentiment.join(lastfm_users, "user_id", "inner")

# F. Final Trends
result_df = final_df.groupBy("country", "gender") \
    .agg(
        F.avg("user_avg_sentiment").alias("avg_sentiment"),
        F.count("user_id").alias("num_users")
    ) \
    .sort(F.col("num_users").desc())

print("Sentiment Trends by Country and Gender:")
result_df.show(20)

# Save to HDFS with Overwrite mode
# output_path = "/user/s2996499/output_sentiment_trends.csv"
# result_df.write.mode("overwrite").csv(output_path, header=True)
