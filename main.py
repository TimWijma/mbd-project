from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.functions import col, sum as sum_, countDistinct

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
    .option("inferSchema", "true") \
    .csv("/data/doina/OSCD-MillionSongDataset/output_*.csv")

# Number of songs each artist has in the MSD
msd_artist_song_counts = (
    df_msd
    .filter(col("artist_mbid").isNotNull())
    .groupBy("artist_mbid")
    .agg(countDistinct("song_id").alias("num_songs"))
)

# Artists with 3 or more songs
msd_artists_3plus = msd_artist_song_counts.filter(col("num_songs") >= 3)

# Artists with 3 or more songs that appear in the lastfm dataset
#artists_with_3_songs_in_both = (
#    lastfm_artists
#    .join(msd_artists_3plus, on="artist_mbid", how="inner")
#)

