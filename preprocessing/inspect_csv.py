import pandas as pd

CSV_PATH = "C:/Users/advit/Documents/convolution neural network/satellite/Global_Landslide_Catalog_Export.csv"

df = pd.read_csv(CSV_PATH)

df["event_date_parsed"] = pd.to_datetime(
    df["event_date"],
    errors="coerce"
)

# Himachal Pradesh candidates
hp = df[
    df["admin_division_name"]
    .astype(str)
    .str.contains("Himachal", case=False, na=False)
].copy()

# 2016 candidates
events_2016 = df[
    df["event_date_parsed"].dt.year == 2016
].copy()

hp_2016 = events_2016[
    events_2016["admin_division_name"]
    .astype(str)
    .str.contains("Himachal", case=False, na=False)
].copy()

print("=" * 80)
print("GLC DATE DIAGNOSTIC")
print("=" * 80)

print("Overall date range:")
print(
    df["event_date_parsed"].min(),
    "to",
    df["event_date_parsed"].max()
)

print()
print("Rows by year:")
print(
    df["event_date_parsed"]
    .dt.year
    .value_counts()
    .sort_index()
)

print()
print("2016 events:", len(events_2016))
print("Himachal events:", len(hp))
print("Himachal 2016 events:", len(hp_2016))

print()
print("=" * 80)
print("HIMACHAL 2016 EVENTS")
print("=" * 80)

if len(hp_2016) > 0:
    print(
        hp_2016[
            [
                "event_id",
                "event_date",
                "event_title",
                "location_description",
                "admin_division_name",
                "longitude",
                "latitude"
            ]
        ].to_string(index=False)
    )
else:
    print("NO HIMACHAL 2016 EVENTS FOUND")

print()
print("=" * 80)