from utils import *

beans_df=pd.read_excel("Dry_Bean_Dataset.xlsx")
beans_df=remove_nulls("Class",beans_df)
beans_df=normalize_dataframe(beans_df)
x_train,y_train,x_test,y_test=split("Class",df=beans_df)
y_train_encoded=one_hot_encode_target(y_train)
y_test_encoded=one_hot_encode_target(y_test)