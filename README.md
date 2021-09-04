# COVID - 19 DATA ANALYSIS GERMANY [View](https://iamswati.github.io/covid_19_project/)

The following repository contains the Note book dealing with Covid 19 data analysis for the country Germany.


## OBJECTIVE

* The main goal of this note book is the analysise and predict the number of new cases for the country germnay in future days.
* Obtain data insights using pandas.
* Cleaning the data with appropriate techniques.
* Performing epxloratory data analysis (EDA) on the data to get better insights.
* Modeling the data with various model with appropriate feature selection techniques.

```
# Importing required libraries
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
```

## Getting Germany Country data

```
# Filtering to Germany
grouplocation=df_data.groupby(df_data.location)
df_germany=grouplocation.get_group("Germany")
pd.set_option('display.max_columns', None)
```

## Checking Skewness Graphically

![2ba98b21-8169-4d66-8927-caaeab4bce66](https://user-images.githubusercontent.com/67102886/132090491-60e2f4ae-5bca-41b1-9321-834e811effc6.png)
![6361af52-3954-4f16-a59d-1d5d64e4faf4](https://user-images.githubusercontent.com/67102886/132090548-338e0b5e-5680-4b47-ac5a-19c5cc83fc5c.png)
![7d9a3630-3e60-457e-8e3c-94c803187252](https://user-images.githubusercontent.com/67102886/132090622-385cba31-3529-468e-8ac8-dd4b99173af5.png)
![f80d92be-ac92-4fa2-9dfc-62930197722b](https://user-images.githubusercontent.com/67102886/132090649-30c34925-b388-49b2-854f-f4f6309c572e.png)
![eed683af-eb1c-4714-963c-dc6713067f6f](https://user-images.githubusercontent.com/67102886/132090629-fdafba6e-d62c-42ce-91ef-e8c03059ec85.png)
![a25afd88-d74c-4196-bec0-b7bd7be34e75](https://user-images.githubusercontent.com/67102886/132090597-fc7a6319-e3e1-4e16-870a-28f86c39aa81.png)
![dd514b00-403a-45e8-ab47-2b916476ffa6](https://user-images.githubusercontent.com/67102886/132090639-a6d11a3e-ffd7-4cb2-8455-94b32e102302.png)
![e3795e67-091d-44f5-8a20-fa0e81ce4539](https://user-images.githubusercontent.com/67102886/132090626-48f8fdcc-9dab-4573-9a7a-e072e3b445d0.png)
![53f05013-e700-43e1-bac1-82e123065907](https://user-images.githubusercontent.com/67102886/132090643-cf7cfe4b-5128-4488-9c39-711b51b34637.png)
![68b26f06-9ce1-46e9-8b8a-fb4515f42221](https://user-images.githubusercontent.com/67102886/132091047-86186faf-bd26-4940-a0bf-d78de41d0d3f.png)
![dfb30a4c-78a9-45ee-855a-7905e1aa3bd2](https://user-images.githubusercontent.com/67102886/132091054-f36bdaff-9b00-455c-81da-268acabf6cbc.png)
![ffde7dcd-d89f-4243-be72-b541c629f5ea](https://user-images.githubusercontent.com/67102886/132091061-43c9b891-47d1-44de-97de-431c6d824e8d.png)
![5cd8ff02-13fb-400e-bac6-8858b746e651](https://user-images.githubusercontent.com/67102886/132091065-932411e8-8abd-4aaf-bdc0-42948d5e02bd.png)
![44e5322e-b87c-4bc1-891d-c8a9d2795270](https://user-images.githubusercontent.com/67102886/132091068-8a456b2b-a75b-457b-9f02-3a599d892d31.png)
![7ef31a4a-7241-475f-8730-bddaf9750ef6](https://user-images.githubusercontent.com/67102886/132091202-7eab2236-9c5e-44f9-8a27-84d7203c4422.png)
![145e885c-cd20-4b68-9635-adc03883c2cb](https://user-images.githubusercontent.com/67102886/132091124-c19e9dd4-7818-4f70-a0a1-8b367688d676.png)
![9f5bd9cb-3c36-43d1-8eee-d605b45d8d5e](https://user-images.githubusercontent.com/67102886/132091127-0f8543e8-aee6-47c3-98ea-e5c6de423bb5.png)
![94dc4f44-bdb7-4f2d-8482-7d1de1033698](https://user-images.githubusercontent.com/67102886/132090766-334b7990-3751-4dcb-a4e2-7d3d94780fd9.png)
![78558ac3-f7a5-4518-972f-5c6a35403ce6](https://user-images.githubusercontent.com/67102886/132091270-23328c0e-67c5-427e-bcb5-6a470cee64b8.png)
![f880a23a-a127-4680-bbf9-0f278f1d0cd6](https://user-images.githubusercontent.com/67102886/132091272-5ea11f17-f233-440c-b3dd-cd01f77b0fa9.png)
![76fac49d-bf7b-4f3c-b380-29093d91c936](https://user-images.githubusercontent.com/67102886/132091279-09eb811f-77e6-443e-bfd1-45582d264036.png)
![ce6678c1-e916-4274-adbd-cd0f1a2ede08](https://user-images.githubusercontent.com/67102886/132091282-d8a5692b-4a61-4c45-92bf-06a77d5807d5.png)
![24c2312e-d4c0-4aa7-bf57-bcc2265b696f](https://user-images.githubusercontent.com/67102886/132090792-c6165f03-2951-4e61-aaaa-4378e4bf1c6c.png)
![be08e9b1-2153-4b9c-93dc-4d0d76bf26ca](https://user-images.githubusercontent.com/67102886/132090802-fa890067-cc73-4a1a-b340-fd194d849606.png)
![a0ff80a7-01bc-410b-aca7-88739707d5e5](https://user-images.githubusercontent.com/67102886/132090805-32772790-bfe2-499c-9e4d-a081972c6746.png)
![4604f29d-9356-4256-9f82-b90093fb9a19](https://user-images.githubusercontent.com/67102886/132090808-be295ef1-5a84-4de5-a3a6-84243fbfb2a9.png)
![dbbdf1af-d802-4481-af9f-8175447d1761](https://user-images.githubusercontent.com/67102886/132090812-fc8479ff-003e-449b-adcc-000c7ebba412.png)
![1c099c38-4b85-4d50-af9f-0b948128ba97](https://user-images.githubusercontent.com/67102886/132090819-d885d827-7075-4d47-b060-aa19285c32cb.png)
![06b522d8-b727-4d70-80d3-4cfe6b37297f](https://user-images.githubusercontent.com/67102886/132090820-5aa2e611-7e77-4c39-a4ca-2b804b8bb6cd.png)

## Total Number of Recovered Cases

![6ae14b37-80e5-400c-affe-c56de189a276](https://user-images.githubusercontent.com/67102886/132091409-11e92b53-7c7b-4409-b87c-9c438f33ad05.png)

![b8602bcd-63b1-4ca9-b5f0-282cf1d31d31](https://user-images.githubusercontent.com/67102886/132091413-14ff6d82-8336-47f3-89b1-3f3092c5a3de.png)

![793330a7-e6e7-4a50-9ba5-d9280f3a44f4](https://user-images.githubusercontent.com/67102886/132091414-9d45c291-1194-4800-9793-a2ce5a7217fe.png)

## Total Number of Active Cases
![f1de6484-fd25-4857-88cd-272dcb48f591](https://user-images.githubusercontent.com/67102886/132091417-981dbf6e-525f-43d2-854e-a2ff4afe7540.png)

![3d489bc9-a85f-4e79-83b1-51f5524297e9](https://user-images.githubusercontent.com/67102886/132091420-bb32d93a-46a3-450a-bee3-daeaf74113a8.png)

## Total Number of New Deaths
![938400fb-86b7-41ad-aae4-15dca44d96ff](https://user-images.githubusercontent.com/67102886/132091428-76b4cb7e-2258-42c1-800f-60d40ede61ad.png)

## Correlation

![Output](https://user-images.githubusercontent.com/67102886/132091432-b36c68c9-8811-4865-b48f-f4dc62c0fae4.png)

## Line Plot gives the relation between how the each column vaires with respect to each Day. 

![5385d8b9-0f1e-4136-b363-6572f0f361b7](https://user-images.githubusercontent.com/67102886/132091760-c542759b-25b0-4c11-9f24-8b36d55f405a.png)
![d2e880ee-8c8f-4d37-b7a4-559bddebdb6d](https://user-images.githubusercontent.com/67102886/132091762-c87c2d52-94f3-4fe9-b7fe-d8ff55fcf2b7.png)
![362d1d2d-6377-4867-8810-2ab34b8e5d5c](https://user-images.githubusercontent.com/67102886/132091765-6b5cc966-7e98-4f3b-942e-f076e6e23314.png)
![0cefa159-6bf2-4869-baa4-9f0726fd529e](https://user-images.githubusercontent.com/67102886/132091767-3a3923e0-014a-439c-b936-c2bc1a2d5f34.png)
![b351d79a-a213-4793-9c2c-97dc90a2f72c](https://user-images.githubusercontent.com/67102886/132091768-a6731dda-5ea8-451e-b930-6f4cbfb867ef.png)
![90330589-b8e4-4c2a-8781-410cfe91dbea](https://user-images.githubusercontent.com/67102886/132091787-5a1e6773-074c-4507-8550-405c732cd482.png)
![c45a1435-d7b4-4d4e-b102-292415a53ec3](https://user-images.githubusercontent.com/67102886/132091789-fabb8c85-5eba-4fd7-85ae-332e3015385c.png)
![f7fb8b58-34f1-4b28-a24b-0d47f822b448](https://user-images.githubusercontent.com/67102886/132091792-ca000510-dfdf-4b23-8268-e24fb704a55e.png)
![b4dbe122-34a1-4f01-b183-2756f81311f3](https://user-images.githubusercontent.com/67102886/132091795-963bdd3e-95b8-4deb-80e0-4b3277ba99e1.png)
![7100ccf5-0438-4df4-a347-0098683e9679](https://user-images.githubusercontent.com/67102886/132091798-f6ae64cf-3baf-4f4d-8e4a-9cabd9305ada.png)
![af9c78e7-4459-48a3-9e74-fef5cf16b98e](https://user-images.githubusercontent.com/67102886/132091819-8ea78069-0e26-4a77-b834-210508a18fea.png)
![9046ed84-a96f-48fe-ad78-a8fbb29f24d0](https://user-images.githubusercontent.com/67102886/132091821-70ebf65f-9cb6-412f-a6a1-f55c21b04e2e.png)
![ff577dd2-46e4-4d31-a8f0-837263e71e99](https://user-images.githubusercontent.com/67102886/132091822-7e585ada-7d19-4cdc-b795-c3abbddecc20.png)
![d338864a-13fd-4ac2-8a48-d747aa97ba40](https://user-images.githubusercontent.com/67102886/132091826-65a929f4-2633-4e35-a573-76c919081ce9.png)
![f2e409ef-0abd-44f0-839b-9cb2e8eed8e7](https://user-images.githubusercontent.com/67102886/132091827-79f413b6-3bcf-4d4a-a7a6-bd03905c16a9.png)
![849fe06f-80e9-4b5a-be0c-67955acca559](https://user-images.githubusercontent.com/67102886/132091830-e8c08148-afac-4696-8e8a-c453050772ba.png)
![79abbc8a-e4e2-4651-9fc6-b720fb403cc7](https://user-images.githubusercontent.com/67102886/132091833-1c38e7ca-4c9b-4274-a5e1-ddd84ec8d5fa.png)
![b0b5923d-5b59-4d33-ad06-aab1ed84e871](https://user-images.githubusercontent.com/67102886/132091834-be5ccdf5-d3ae-4b14-b6ad-debc0e127967.png)
![8db0d3cf-b665-4b17-8c12-9e74b98ecd92](https://user-images.githubusercontent.com/67102886/132091838-2b8b45b4-cbd7-4986-ae3d-3490e7bb4c3d.png)
![9fbec232-c3d9-49e1-96a5-b5551bef9240](https://user-images.githubusercontent.com/67102886/132091840-bd620f05-d644-49a6-abce-1fb2c188bc66.png)
![75a4551f-ab1f-47cf-baf8-8ad330dac6aa](https://user-images.githubusercontent.com/67102886/132091843-c726640c-c090-467b-a96e-dbe0b0a98eb7.png)
![bcc0fc56-b09b-4253-949f-86dddbcc93d0](https://user-images.githubusercontent.com/67102886/132091849-2288a72b-88b2-49dd-b964-ee5b22ddbc9e.png)
![1dc22f3f-c7c8-4fde-b641-a95f0d147770](https://user-images.githubusercontent.com/67102886/132091852-feeff463-c395-45d8-9f46-5d609d6597cb.png)
![855cc624-2d0d-4b31-9873-3ebd023b262a](https://user-images.githubusercontent.com/67102886/132091853-fd71a99a-e00c-4f51-b742-0311ca8e5a20.png)
![a3817a98-40f2-4294-acc1-69727ac3e647](https://user-images.githubusercontent.com/67102886/132091854-e9a6a19b-905b-442a-9258-82e935911b6e.png)
![7676da8b-ed69-4a44-b392-7ce6c27189d1](https://user-images.githubusercontent.com/67102886/132091856-bc65e2c7-fddb-4ca5-926b-96ad65ad6800.png)
