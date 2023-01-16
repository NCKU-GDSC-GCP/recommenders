import pandas as pd
import numpy as np


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        data_train_full = pd.read_csv("dataset/train_full.csv", low_memory=False)
        data_orders = pd.read_csv("dataset/orders.csv", low_memory=False)
        
        dataset1 = data_train_full[["customer_id", "gender", "location_type", "id", "OpeningTime", "language", "vendor_rating", "serving_distance", "vendor_tag_name", "delivery_charge"]]
        dataset1.rename(columns={"vendor_rating": "mean_rating"}, inplace=True)
        cols = ["customer_id", "id"]
        dataset1["all"] = dataset1[cols].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
        dataset1.drop_duplicates(["all"], inplace=True)

        dataset2 = data_orders[["akeed_order_id","customer_id","vendor_id", "item_count", "grand_total", "vendor_rating"]]
        dataset2.rename(columns={"vendor_id": "id"}, inplace=True)
        cols = ["customer_id", "id"]
        dataset2["all"] = dataset2[cols].apply(lambda row: "_".join(row.values.astype(str)), axis=1)

        df = pd.merge(dataset1, dataset2, on="all", how="inner")
        df.rename(columns={"customer_id_x": "customer_id"}, inplace=True)
        df.rename(columns={"id_x": "vendor_id"}, inplace=True)
        df.drop(["customer_id_y", "id_y"],axis=1, inplace=True)
        #Drop language columns
        df.drop(["language"], axis=1, inplace=True)
        #Remove null of gender columns
        df = df[df["gender"].notnull()].reset_index(drop=True)
        sex = pd.get_dummies(df["gender"], columns=["gender"], prefix="sex", drop_first=True)
        df = pd.concat([df, sex], axis=1)
        df.drop(["gender"], axis=1, inplace=True)
        df.rename(columns={"vendor_rating": "rating"}, inplace=True)

        ratings = df[["customer_id", "vendor_id", "rating"]]

        ratings["rating2"] = ratings["rating"].apply(self.rating_missing_func)
        ratings = ratings[["customer_id", "vendor_id", "rating2"]]
        ratings.rename(columns={"rating2": "rating", 1: "customer_id_num"}, inplace=True)
        ratings = ratings.groupby(["customer_id", "vendor_id"]).mean().reset_index()
        R_temp = ratings.pivot(index="customer_id", columns="vendor_id", values="rating").fillna(0)

        customer_id_index = []

        for i, one_id in enumerate(R_temp.T):
            customer_id_index.append([one_id, i])
        df_customer_id_index = pd.DataFrame(customer_id_index)
        df_customer_id_index.rename(columns={0:"customer_id", 1:"customer_idx"}, inplace=True)

        vendor_id_index = []

        for i, one_id in enumerate(R_temp) :
            vendor_id_index.append([one_id, i])
        df_vendor_id_index = pd.DataFrame(vendor_id_index)
        df_vendor_id_index.rename(columns={0:"vendor_id", 1:"vendor_idx"}, inplace=True)

        ratings_with_index = pd.merge(ratings, df_customer_id_index, on="customer_id")
        ratings_with_index = pd.merge(ratings_with_index, df_vendor_id_index, on="vendor_id")
        ratings = ratings_with_index[["customer_idx", "vendor_idx", "rating"]].astype(int)
        ratings.rename(columns={"customer_id_num":"customer_idx", "vendor_id_num":"vendor_idx", "rating": "rating"}, inplace=True)
        self.data = ratings


    def rating_missing_func(x):
        if pd.isnull(x) == True:
            return 0
        elif x == 0:
            return 0
        else:
            return x
