# ============================ libraries =====================================
import numpy as np

### ========================== functions =====================================

def test_year_quarter_by_park(park):
    if park == "Gonarezhou" or park == "mbe" or park == "mbe_gun" or park == "CRNP":
        return 2018, 1
    elif park == "AMWS":
        return 2017, 4
    elif park == "MFNP" or park == "QENP":
        return 2015, None
    elif park == "Mondulkiri":
        return 2017, 4
    elif park == "Dja":
        return 2018, 1
    elif park == "MPF":
        return 2017, 1
    elif park == "SWS":
        return 2018, 1
    else:
        raise Exception("Park '{}' not implemented.".format(park))

# TODO: kai says this may not be used anymore
def selected_threshold_by_park(park):
    if park == "QENP":
        return np.arange(0, 8, 0.5)
    elif park == "MFNP":
        return np.arange(0, 8, 0.5)
    elif park == "CRNP":
        return np.arange(0, 1.2, 0.1)
    elif park == "Gonarezhou":
        return np.arange(0, 15, 1.5)
    elif park == "AMWS":
        return np.arange(0, 1.2, 0.05)
    elif park == "mbe" or park == "mbe_gun":
        return np.arange(0, 1.65, 0.15)
    # elif park == "Mondulkiri":
    #     return np.arange(0, 12, 0.5)
    elif park == "Dja":
        return np.arange(0, 7, 0.5)
    elif park == "MPF":
        #return np.arange(0, 15, 1.5)
        #return np.arange(0, 7, 0.5)
        # return np.arange(0, 12, 0.5)  # for full data
        return np.arange(0, 5, 0.3)# for rainy/dry data
    elif park == "SWS":
        return np.arange(0, 12, 0.5)  # for full data
    else:
        raise Exception("Park '{}' not implemented.".format(park))

# used to predict on static data
def selected_finer_threshold_by_park(park):
    if park == "QENP":
        return np.arange(0, 0.5, 0.1)
    elif park == "MFNP":
        return np.arange(0, 0.5, 0.1)
    elif park == "CRNP":
        return np.arange(0, 0.5, 0.1)
    elif park == "Gonarezhou":
        return np.arange(0, 1.5, 0.5)
    elif park == "AMWS":
        return np.arange(0, 0.5, 0.1)
    elif park == "mbe" or park == "mbe_gun":
        return np.arange(0, 1.65, 0.15)
    # elif park == "Mondulkiri":
    #     return np.arange(0, 0.5, 0.1)
    elif park == "Dja":
        return np.arange(0, 1.5, 0.5)
    elif park == "MPF":
        # return np.arange(0, 1.5, 0.5)
        return np.arange(0, 0.5, 0.1)
    elif park == "SWS":
        return np.arange(0, 1.5, 0.5)
    else:
        raise Exception("Park '{}' not implemented.".format(park))


# def illegalActivityTypeList_by_park(park):
#     if park == "QENP":
#         return ['Animal_Com', 'Animal_NonCom', 'Plant_NonCom', 'Plant_Com','Fishing', 'Encroachment']
#     elif park == "MFNP":
#         return ['Animal_Com', 'Animal_NonCom', 'Plant_NonCom', 'Plant_Com','Fishing', 'Encroachment']
#     #elif park == "MPF":
#     #    return ['Trap','Firearms', 'Death_Animals']
#     elif park == "Gonarezhou" or park == "AMWS" or park == "CRNP" or park == "mbe":
#         return ['Illegal_Activity']
#     elif park == "Dja" or park == "MPF":
#         return ["Illegal_Activity"]
#     else:
#         raise Exception("Park '{}' not implemented.".format(park))


# def indicatorList_by_park(park):
#     if park == "QENP" or park == "MFNP":
#         return ["ID-Global", 'Year', 'Month', 'ID-Spatial', "x", "y"]
#     elif park == "Gonarezhou" or park == "AMWS" or park == "CRNP" or park == "mbe":
#         return ["ID_Global", 'Year', 'Quarter', 'ID_Spatial', "x", "y"]
#     elif park == "Dja" or park == "MPF":
#         return ["ID_Global", "Year", "Quarter", "ID_Spatial", "x", "y"]
#     else:
#         raise Exception("Park '{}' not implemented.".format(park))

# def global_ID_format_by_park(park):
#     if park == "QENP" or park == "MFNP":
#         return "ID-Global"
#     elif park == "Gonarezhou" or park == "AMWS" or park == "CRNP" or park == "mbe":
#         return "ID_Global"
#     elif park == "Dja" or park == "MPF":
#         return "ID_Global"
#     else:
#         raise Exception("Park '{}' not implemented.".format(park))



# TODO: may not be used? not really matter
# def train_length_by_park(park):
#     if park == "QENP":
#         return 6
#     elif park == "MFNP":
#         return 6
#     elif park == "CRNP":
#         return 3
#     elif park == "Gonarezhou":
#         return 3
#     elif park == "AMWS":
#         return 7
#     elif park == "mbe" or park == "mbe_gun":
#         return 2
#     elif park == "Mondulkiri":
#         return 4
#     elif park == "Dja":
#         return 3
#     else:
#         raise Exception("Park '{}' not implemented.".format(park))

# def feature_options_by_park(park):
#     if park == "Gonarezhou":
#         return ["Water_Distance", "River_Distance", "Road_Distance", "Boundary_Distance"]
#     elif park == "mbe":
#         return ["Water_Distance", "Water_Distance2", "Road_Distance", "Boundary_Distance", "Habitat"]
#     elif park == "CRNP":
#         return ["Water_Distance", "Water_Distance2", "Road_Distance", "Boundary_Distance", "Habitat"]
#     elif park == "AMWS":
#         return ["Water_Distance", "Water_Distance2", "Road_Distance", "Boundary_Distance", "Habitat"]
#     elif park == "Dja":
#         return ["Water_Distance", "Water_Distance2", "Road_Distance", "Boundary_Distance", "Habitat"]
#         # return ["road", "Water_Distance", "River_Distance", "Habitat_multipleClasses", "boundary"]
#     elif park == "MPF":
#         return ["Water_Distance", "Road_Distance", "Boundary_Distance", ]
#     else:
#         raise Exception("Park '{}' not implemented.".format(park))
