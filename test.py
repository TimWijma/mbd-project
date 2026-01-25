from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.functions import col, sum as sum_, countDistinct, split, slice as slice_func, expr, size

spark = SparkSession.builder.appName("lastfm").getOrCreate()

user_schema = StructType() \
	.add("user_id", StringType(), True) \
	.add("gender", StringType(), True) \
	.add("age", IntegerType(), True) \
	.add("country", StringType(), True) \
	.add("signup", StringType(), True) \

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

lastfm_user_artists = (
        spark.read
        .option("delimiter", "\t")
        .schema(artist_schema)
        .csv("/user/s2996499/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv")
)

# All unique artists (with a musicbrainz id) in the lastfm dataset
lastfm_artists = (
    lastfm_user_artists
    .select("artist_mbid")
    .filter(col("artist_mbid").isNotNull())
    .distinct()
)

df_msd = spark.read \
    .option("header", "true") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .option("multiLine", "true") \
    .csv("/data/doina/OSCD-MillionSongDataset/output_*.csv")

df_msd_bridge = df_msd \
    .select(
        col("track_id").alias("msd_track_id"), 
        col("artist_mbid").alias("msd_artist_mbid")
    )




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

df_mxm = df_parsed.filter(size(col("word_counts_raw")) > 0)

df_lastfm_unique = lastfm_user_artists \
    .select("artist_mbid") \
    .filter(col("artist_mbid").isNotNull()) \
    .distinct()


artists_with_lyrics = df_mxm.join(
    df_msd_bridge,
    df_mxm.msd_track_id == df_msd_bridge.msd_track_id,
    "inner"
).select(col("msd_artist_mbid").alias("artist_mbid")).distinct()

valid_artists_in_lastfm = df_lastfm_unique.join(
    artists_with_lyrics,
    on="artist_mbid",
    how="inner"
)

total_lastfm_artists = df_lastfm_unique.count()
total_lyric_artists = artists_with_lyrics.count()
overlap_count = valid_artists_in_lastfm.count()

print(f"Total Unique Artists in Last.fm 360k: {total_lastfm_artists}")
print(f"Total Artists with Lyrics in Musixmatch: {total_lyric_artists}")
print(f"MATCH: Artists in Last.fm who have Lyrics: {overlap_count}")

if total_lastfm_artists > 0:
    print(f"Coverage: {(overlap_count / total_lastfm_artists) * 100:.2f}%")
