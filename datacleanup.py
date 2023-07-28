import numpy as np

def getRawData(file):
    f = open(file, 'r')
    data = f.readlines()
    f.close()
    return data

def getSpreadsheet(data):
    spreadsheet = []
    for datum in data:
        datum = datum.split()
        spreadsheet.append(datum)
    return spreadsheet

def getIdsAndQuizzes(spreadsheet):
    ids = []
    quizzes = []
    for i in spreadsheet:
        ids.append(i[0])
        quizzes.append(i[1])
    return ids, quizzes

def countOccurs(val, liste):
    occur = 0
    for i in liste:
        if (i == val):
            occur += 1
    return occur
        
def applyThresholds1(data):
    
    final_data = []
    
    for datum in data:
        if (datum[0] < 5 and datum[1] < 1.1 and datum[2] < 60 and datum[3] < 60 and datum[5] < 30 and datum[6] < 25):
            final_data.append(datum)
            
    final_data = np.array(final_data, dtype=float)
            
    return final_data
    
def applyThresholds2(data):
    
    final_data = []
    
    for datum in data:
        if (datum[2] < 5 and datum[3] < 1.1 and datum[5] < 60 and datum[6] < 60 and datum[9] < 30 and datum[10] < 25):
            final_data.append(datum)
            
    final_data = np.array(final_data)
            
    return final_data

def applyThresholds3(data):
    
    final_data = []
    
    for datum in data:
        if (datum[1] < 5 and datum[2] < 1.1 and datum[4] < 60 and datum[5] < 60 and datum[8] < 30 and datum[9] < 25):
            final_data.append(datum)
            
    final_data = np.array(final_data)
            
    return final_data

def test(data, ids):
    
    final_data = []
    new_ids = []
    i = 0
    
    for datum in data:
        if (datum[1] < 5 and datum[2] < 1.1 and datum[4] < 60 and datum[5] < 60 and datum[8] < 30 and datum[9] < 25):
            final_data.append(datum)
            new_ids.append(ids[i])
        i += 1
    
    final_data = np.array(final_data)
            
    return final_data, new_ids

def getFinalData1():
    raw_data = getRawData('behavior-performance.txt')
    spreadsheet = getSpreadsheet(raw_data)
    ids, quizzes = getIdsAndQuizzes(spreadsheet)
    
    final_data = []
    for i in spreadsheet:
        if (countOccurs(i[0] , ids) >= 5): #depending on question, change required_num to minimum quizzes to complete by students
            final_data.append(i)
    
    final_data = np.array(final_data)
    final_data = final_data.transpose()
    final_data = np.delete(final_data, [0,1,4,8,11], axis=0)
    final_data = np.array(final_data, dtype=float)
    final_data = final_data.transpose()
    final_data = applyThresholds1(final_data)
    
    return final_data
 
def getFinalData2():
    raw_data = getRawData('behavior-performance.txt')
    spreadsheet = getSpreadsheet(raw_data)
    ids, quizzes = getIdsAndQuizzes(spreadsheet)
    
    number_quizzes = len(set(quizzes)) - 1 #The minus sign accounts for 'VidID'. There are 92 quizzes in total
    required_num = int(number_quizzes / 2)
    
    final_data = []
    for i in spreadsheet:
        if (countOccurs(i[0] , ids) >= required_num): #depending on question, change required_num to minimum quizzes to complete by students
            final_data.append(i)
    
    final_data = np.array(final_data)
    final_data = final_data.transpose()
    ids = final_data[0, :]
    final_data = np.delete(final_data, [0], axis=0)
    final_data = np.array(final_data, dtype=float)
    final_data = final_data.transpose()
    final_data = test(final_data, ids)
    
    ids = final_data[1]
    final_data = final_data[0]
    
    return final_data, ids
    
def getFinalData3():
    raw_data = getRawData('behavior-performance.txt')
    spreadsheet = getSpreadsheet(raw_data)
    
    final_data = spreadsheet[1::]
    final_data = np.array(final_data)
    final_data = final_data.transpose()
    final_data = np.delete(final_data, [0], axis=0)
    final_data = np.array(final_data, dtype=float)
    final_data = final_data.transpose()
    final_data = applyThresholds3(final_data)
          
    return final_data

def getIdsAndScores():
    data, ids = getFinalData2()
    
    scores = []
    
    for i in data:
        scores.append(i[10])

    id_list = set(ids)
    avg_list = []
    avg_features = []

    for idd in id_list:
        i = 0
        temp = []
        temp2 = []
        for idd2 in ids:
            if idd == idd2:
                temp.append(scores[i])
                temp2.append(data[i,1:10])
            i += 1
        avg_features.append(np.mean(temp2, axis=0))
        avg_list.append(sum(temp)/len(temp))
    
    return id_list, avg_list, avg_features