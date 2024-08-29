import pandas as pd
import numpy as np

size = 20

heights = np.random.randint(150, 200, size)
colors = np.random.choice(
    ["zielone", "niebieskie", "brazowe"], size=size, p=[0.3, 0.3, 0.4]
)

data = pd.DataFrame({"wzrost": heights, "kolor_oczu": colors})

data_one_hot = pd.get_dummies(data=data.copy(), columns=["kolor_oczu"], prefix="oczy")
print(data_one_hot)
data_one_hot_dropped = pd.get_dummies(
    data=data.copy(), columns=["kolor_oczu"], prefix="oczy", drop_first=True
)
print(data_one_hot_dropped)
