from langdetect import detect
# import langid
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from textblob import TextBlob


class ExcelIO:
    def __init__(self, folder_path=None, df=None, df_density=None):
        self.folder_path = folder_path
        self.df = df
        self.df_density = df_density

    def import_csv(self):
        self.folder_path = os.getcwd()
        self.df = pd.read_csv(self.folder_path + '\\listings.csv')
        self.df_density = pd.read_csv(self.folder_path + '\\neighbourhood_density.csv')


excel = ExcelIO()
excel.import_csv()


class DFMiner:
    def __init__(self, df=None, df_density=None):
        self.df = df
        self.df_density = df_density

    # ===== id ===== #
    # 1. by number of reviews = 1 and last_review date, we may guess the post date, so it may affect the price due the inflation

    # ===== name ===== #
    # v. name can disclose room type, only 20% can be identified
    def get_room_count(self, x):
        # remove special character. Add / at the end to prevent null string case
        x = ''.join(e for e in str(x).upper().replace(' ', '') if e.isalnum()) + '/'
        # make sure the result are (digit) + BR pattern
        room_type_map = {'PRIVATEROOM': x[max(x.find('PRIVATEROOM')-1, 0)] if x[max(x.find('PRIVATEROOM')-1, 0)].isdigit() else "NA",
                         'DOUBLE': x[max(x.find('DOUBLE') - 1, 0)] if x[max(x.find('DOUBLE') - 1, 0)].isdigit() else "NA",
                         'BD': x[max(x.find('BD') - 1, 0)] if x[max(x.find('BD') - 1, 0)].isdigit() else "NA",
                         'BED': x[max(x.find('BED') - 1, 0)] if x[max(x.find('BED') - 1, 0)].isdigit() else "NA",
                         'BR': x[max(x.find('BR') - 1, 0)] if x[max(x.find('BR') - 1, 0)].isdigit() else "NA",
                         'ROOM': x[max(x.find('ROOM')-1, 0)] if x[max(x.find('ROOM')-1, 0)].isdigit() else "NA",
                         '房': x[max(x.find('房') - 1, 0)] if x[max(x.find('房') - 1, 0)].isdigit() else "NA",
                         'DORM': str(0.25),  # Assume 4 Beds in Dorm
                         }
        for key, value in room_type_map.items():
            if key in str(x):
                return value
        return "NA"

    # v. name can disclose near area, there is around 30% name turned to array.
    def get_near_loc(self, x):
        name_lst = re.sub('[^A-Za-z /@]+', '', str(x).upper()).split(' ')
        near_loc_lst = {'NEXT', 'NEAR', '@', 'AT', 'WALK'}

        for key in near_loc_lst:
            if key in name_lst:
                try:
                    if name_lst[name_lst.index(key)+1] in ['THE', 'TO']:
                        return [y for y in name_lst[name_lst.index(key)+2].split('/') if y]
                    else:
                        return [y for y in name_lst[name_lst.index(key)+1].split('/') if y]
                except:
                    return ["NA"]
        return ["NA"]

    # v. name can disclose transportation (e.g. MRT/ Metro/ Airport/ Train/ Station)
    def get_near_metro(self, x):
        x = str(x).upper()
        return x.count('MRT') + x.count('METRO') + + x.count('TRAIN') + x.count('STATION') > 0

    def get_near_airport(self, x):
        x = str(x).upper()
        return x.count('AIRPORT') > 0

    # v. wifi provided
    def get_wifi(self, x):
        x = str(x).upper()
        return x.count('WIFI') > 0

    # ===== host_name ===== #
    # *. by host_name, we may guess what is the ethics of the host. (Pending in this moment, since it is out of expectation)
    def get_host_lang(self, x):
        x = ''.join(e for e in str(x) if e.isalpha())
        try:
            # return langid.classify(x)[0]
            # x = re.sub(r'[^a-zA-Z ]', '', x)
            return detect(str(x))
            # return TextBlob(str(x)).detect_language()
        except:
            return "NA"

    # ===== neighbourhood_group ===== #
    # 1. It may link to extend database about the environment of that region
    def get_density_except_central(self):
        self.df = pd.merge(self.df, self.df_density, how='left', left_on=['neighbourhood'], right_on=['Planning'])

    # ===== price ===== #
    # ~. 0 price exists

    # ===== reviews_per_month ===== #
    # * nan need to change to zero.
    def refresh_reviews_per_month_zero(self):
        self.df['reviews_per_month'] = self.df['reviews_per_month'].fillna(0)

    # ===== availability_365 ===== #
    # 1. change to rate
    def get_365_rate(self, x):
        try:
            return int(x)/365
        except ValueError:
            return 0


    def main(self):

        self.df['room_count'] = self.df['name'].apply(self.get_room_count)
        self.df['mention_near'] = self.df['name'].apply(self.get_near_loc)
        self.df['mention_metro'] = self.df['name'].apply(self.get_near_metro)
        self.df['mention_airport'] = self.df['name'].apply(self.get_near_airport)
        self.df['mention_wifi'] = self.df['name'].apply(self.get_wifi)
        # self.df['host_lang'] = self.df['name'].apply(self.get_host_lang)
        # self.df['host_lang2'] = self.df['host_name'].apply(self.get_host_lang)
        self.get_density_except_central()
        self.refresh_reviews_per_month_zero()
        self.df['availability_365'] = self.df['availability_365'].apply(self.get_365_rate)


mined = DFMiner(excel.df.copy(), excel.df_density.copy())
mined.main()


class ChartCreator:
    def __init__(self, df, corr_df=None, corr_heatmap=None, neighbour_plot=None, neighbour2_bar=None, near_transport_bar=None):
        self.df = df
        self.corr_df = corr_df
        self.corr_heatmap = corr_heatmap
        self.neighbour_plot = neighbour_plot
        self.neighbour2_bar = neighbour2_bar
        self.near_transport_bar = near_transport_bar

    # 1. Plot Correlation Heatmap
    def get_correlation_matrix(self):
        self.corr_df = self.df.corr()

    def plot_correlation_heatmap(self):
        # Generate a mask for the upper triangle
        mask = np.zeros_like(self.corr_df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(14, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        self.corr_heatmap = sns.heatmap(self.corr_df,
                                        mask=mask, cmap=cmap, vmax=.5, center=0,
                                        square=True, linewidths=.8, cbar_kws={"shrink": .8},
                                        annot=True, fmt=".2f")
        self.corr_heatmap.figure.text(0.6, 0.9,
                                      "There is no obvious correlation between PRICE\nand other factors in the correlation heatmap")

    # 2. Find closed correlated factor and do the further study
    #   a. Location: Neighbourhood_Group/ Neighbourhood
    def plot_neighbour_gp_against_price(self):
        plt.figure(figsize=(20, 8))
        self.neighbour_plot = sns.boxplot(x='neighbourhood_group', y='price', data=self.df)
        self.neighbour_plot.set_xlabel(xlabel="Planning Area")
        self.neighbour_plot.set_ylabel(ylabel="Price")
        self.neighbour_plot.set_title(label="Planning Area Against Price", fontdict={'size': 20, 'color': 'darkred', 'weight': 'bold'})
        self.neighbour_plot.figure.text(0.4, 0.85, "Central Region have the highest renting price,\n"
                                                   "on the other hand, North-East Region have lowest renting price", ha='left', fontsize=8)
        self.neighbour_plot.figure.get_axes()[0].set_yscale('log')

    def plot_neighbour_against_price(self):
        plt.figure(figsize=(20, 8))
        top10_price_by_neighbourhood = self.df.groupby(['neighbourhood'])['price'].mean().sort_values(ascending=False)[-10:, ]
        self.neighbour2_bar = sns.barplot(x=top10_price_by_neighbourhood.index, y=top10_price_by_neighbourhood.values, alpha=0.8)
        self.neighbour2_bar.set_title(label="Bottom 10 Renting Price In Singapore")
        self.neighbour2_bar.figure.text(0.6, 0.85, "Region near Malaysia, North side of the Singapore,\n"
                                                   "have lower Renting Price", ha='left', fontsize=8)

    #   b. Near Facilities: Near Metro, Near Airplane
    def plot_near_transport(self):
        plt.figure(figsize=(6, 20))
        self.near_transport_bar = sns.violinplot(x='mention_metro', y='price', data=self.df)
        self.near_transport_bar.set_ylim(ymin=0.0, ymax=500.0)
        self.near_transport_bar.figure.text(0.2, 0.9, "Name mentioned metro don't have obvious correlation to the price", ha='left', fontsize=8)

    def main(self):
        # self.get_correlation_matrix()
        # self.plot_correlation_heatmap()
        # self.plot_neighbour_gp_against_price()
        # self.plot_neighbour_against_price()
        self.plot_near_transport()


chart = ChartCreator(mined.df.copy())
chart.main()

# ===== TARGET ===== #
# *. Enhance Business by find the correlation to price
# *. Enhance Business by predict the expected price

# ===== CASE STUDY ===== #
# *. Senior Host Latest id description

# ===== TESTING ===== #
test = mined.df.copy().sort_values(by=['neighbourhood_group'], ascending=[True])
