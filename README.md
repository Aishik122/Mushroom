# iNeuron_Mushroom_Classification
The complete end to end project with deployment on PythonAnyware.
App Link : https://aishik1.pythonanywhere.com/

# ABSTRACT : 
Mushrooms have been consumed since earliest history. The word Mushroom is derived from the French word for Fungi and Mold. Now-a-days, Mushroom are popular valuable food because they are low in calories, carbohydrate, Fat, sodium and also cholesterol free. Besides this, Mushroom provides important nutrients, including selenium, potassium, riboflavin, niacin, Vitamin D, proteins and fiber. All together with a long history as food source. Mushroom are important for their healing capacity and properties in traditional medicine. It has reported beneficial effects for health and treatment of some disease. Many nutraceutical properties are described in Mushroom like cancer and antitumor attributes. Mushroom act as antibacterial, immune system enhancer and cholesterol lowering Agent. Additionally, they are important source of bio-active compounds. This work is a machine learning model that classifies mushrooms into 2 classes: Poisonous and Edible depending on the features of the mushroom. During this machine learning implementation, we are going to see which features are important to predict whether a mushroom is poisonous or edible.

# Problem Statement :
 The Audubon Society Field Guide to North American Mushrooms contains descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom (1981). Each species is labelled as either definitely edible, definitely poisonous, or maybe edible but not recommended. This last category was merged with the toxic category. The Guide asserts unequivocally that there is no simple rule for judging a mushroom's edibility, such as "leaflets three, leave it be" for Poisonous Oak and Ivy.

The main goal is to predict which mushroom is poisonous & which is edible.

# Summary :

* Data authentication: The source of data is iNeuron.ai .The data is good for analysis
* Data bias : After analysing data it looks like data is not bias .
1. The target column has 2 class type one is 'poisonous' which has 3916 counts and second is 'edible' which has 4208 counts so we have nearly equal counts for poisonous and edible classes in our data. Hence we can say that our data is balanced.
2. There are 4 types of cap-surface in a mushroom and also it suggests that 'edible' mushrooms do not have 'cap-surface' : 'g : grooves' according to our data.
3. 51.8 % Mushrooms are Edible.
4. Some people think that all blue bruising mushrooms are safe to eat or are hallucinogenic. The bolete rule above proves that is not true. This myth is an example of why identifying mushrooms through bruising alone is a bad idea.(source google = 'https://www.mushroom-appreciation.com/identifying-mushrooms.html#:~:text=The%20spores%20and%20stem%20turn,alone%20is%20a%20bad%20idea!')
5. 3528 mushrooms dosent have odor
6. cap-shape sunken mushrooms in this dataset is not poisonous in nature where conical is poisonous in nature . other are mixed.
7. Mushrooms with out Bruises have higher chance of being poisonous while with bruises have lower chance being poisonous.
8. mushrooms with almond and anise is edible and no odor is high channce bring edible . Other odor is not recomended for eating
9. abundant and numerous population class are edible according to this data where other are mixed .
10.The 'poisonous' mushrooms do not have Habitat Type as Waste according to this data.
10. stalk color Gray,Orange,Red are completely edible and buff,cinnamon,yellow are poisonous . brown have higher chance of being poisonous .
