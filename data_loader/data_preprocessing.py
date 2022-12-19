"""
remove useless columns
-> collect all orders by every user and sort by time
-> label   h1---h2---h3---h4---h5---h6--->timestamp   h6: label, h1-h5: data
"""


import pandas as pd

def preprocess():
    rawTrain_customers = pd.read_csv("train_customers.csv", low_memory=False)
    train_customers = rawTrain_customers[["akeed_customer_id", "gender"]]
    train_customers.columns = ["user_id", "gender"]


    rawOrders = pd.read_csv("orders.csv", low_memory=False)
    orders = rawOrders[["customer_id", "item_count", "grand_total", "payment_mode",
                         "vendor_id", "created_at", "LOCATION_NUMBER"]]
    orders.columns = ["user_id", "item_count", "cost", "payment_mode",
                        "vendor_id", "time", "location"]
    orders = orders.fillna(1)


    rawVendors = pd.read_csv("vendors.csv")
    vendors = rawVendors[["id", "vendor_category_en", "prepration_time"]]



    train_data = {}
    userMap = {}
    for i in range(len(train_customers)):
        userMap[train_customers["user_id"][i]] = i
        train_data[i] = {}
        train_data[i]["history"] = []
        if train_customers["gender"][i] is None:
            GENDER = 0
        elif train_customers["gender"][i] == "Male":
            GENDER = 1
        else:
            GENDER = 2
        train_data[i]["gender"] = GENDER

    vendorMap = {}
    for i in range(len(vendors)):
        vendorMap[vendors["id"][i]] = i


    def compute_time(time_and_date):
        """ compute the timestamp of the order. """

        time_arr = time_and_date.split(" ")
        date = time_arr[0].split("-")
        date = [int(s) for s in date]
        clock_time = time_arr[1].split(":")
        clock_time = [int(s) for s in clock_time]
        time_stamp = (date[0] - 2019) * 31536000 + date[2] * 86400 +\
                        clock_time[0] * 3600 + clock_time[1] * 60 + clock_time[2]
        if date[1] in [1, 3, 5, 7, 8, 10, 12]:
            time_stamp += date[1] * 2678400
        elif date[1] in [4, 6, 9, 11]:
            time_stamp += date[1] * 2592000
        else:
            time_stamp += 2419200
        return time_stamp


    typeMap = {"Restaurants": 0, "Sweets & Bakes": 1}
    for i in range(len(orders)):
        if orders["user_id"][i] in userMap:
            user_id = userMap[orders["user_id"][i]]
        time = compute_time(orders["time"][i])
        vendor = vendorMap[orders["vendor_id"][i]]
        element = [time, orders["location"][i],
                    vendor, orders["cost"][i],
                    orders["item_count"][i], orders["payment_mode"][i],
                    typeMap[vendors["vendor_category_en"][vendor]],
                    vendors["prepration_time"][vendor]]
        train_data[user_id]["history"].append(element)


    d = []
    for key, value in train_data.items():
        if len(value["history"]) < 11:
            d.append(key)

    for item in d:
        train_data.pop(item)

    for key, value in train_data.items():
        value["history"].sort()


    data = []
    label = []
    for key, value in train_data.items():
        for i in range(len(value["history"])-10):
            tmp = []
            tmp.append(key)
            tmp.append(value["gender"])
            for j in range(11):
                if j != 10:
                    for k in range(len(value["history"][i+j])):
                        tmp.append(value["history"][i+j][k])
                else:
                    label.append(value["history"][i+j][2])
            data.append(tmp)

    datas = (data, label)

    return datas