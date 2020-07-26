import pandas as pd
import pickle

'''bring in the comp builder from the main program and repickle the data to save down that work'''
# get_comps = pickle.load( open( "/home/gregjroberts1/Projects/Data/comp_builder_update.p", "rb" ) )
# # # #pickle a new file subset
# with open('/home/gregjroberts1/Projects/Data/comps470_.p', 'wb') as f:
#       pickle.dump(get_comps, f)


get_final = pickle.load( open( "/home/gregjroberts1/Projects/Data/spx_word_analysis.p", "rb" ) )

get_comps0_100 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps100.p", "rb" ) )
get_comps101_148 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps101_148.p", "rb" ) )
get_comps149_174 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps149_174.p", "rb" ) )
get_comps175_199 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps175_199.p", "rb" ) )
get_comps200_224 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps200_224.p", "rb" ) )
get_comps225_274 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps225_274.p", "rb" ) )
get_comps275_299 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps275_299.p", "rb" ) )
get_comps300_329 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps300_329.p", "rb" ) )
get_comps330_369 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps330_369.p", "rb" ) )
get_comps370_379 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps370_379.p", "rb" ) )
get_comps380_469 = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps380_469.p", "rb" ) )
get_comps470  = pickle.load( open( "/home/gregjroberts1/Projects/Data/comps470_.p", "rb" ) )

get_spx_table = pickle.load( open( "/home/gregjroberts1/Projects/Data/spx_wiki_table.p", "rb" ) )
#Let's update our original dataset for the calculated values

# pd.set_option('display.max_columns', None)
combine_df= pd.concat([get_comps0_100,get_comps101_148,
    get_comps149_174,get_comps175_199,get_comps200_224,
    get_comps225_274,get_comps275_299,get_comps300_329,
    get_comps330_369,get_comps370_379,get_comps380_469,
    get_comps470, get_final],
    ignore_index=True, sort=False)
#
result = pd.concat([get_spx_table, combine_df], axis=1)
# print(combine_df.shape)
# # print(get_final)
print(result.shape)


# print(get_comps470.shape)
# print(get_comps470)
result = result.dropna()
# '''Run our metrics to understand the data better'''
print(result.isna().sum())
print(result.shape)
print(get_spx_table.shape)

print(result.dtypes)

with open('/home/gregjroberts1/Projects/Data/spx_filings_nlp.p', 'wb') as f:
       pickle.dump(result, f)

# #how many of the values have zero value

# zero_values = result[result['YoY Chg']==0]
# print(zero_values[['Security','CIK']].head())

'''ultimately, we're trying to understand if the copany has become riskier with
respect to itself, over time, that relative change in risk is what we're comparing here.
not the absolute level of risk(not a cosine simililarity analysis) but relative
change in how the company addresses it's own admission.  is the market appreciating
this relative update?'''
