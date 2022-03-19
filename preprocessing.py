import os

def remove_lines(text):
    text = text.replace('\r\n', ' ')
    text = text.replace('  ', ' ') #backup double space eliminator :)
    return text

def replace_names(advisor, customer, text):
    text = text.replace('ADVISOR', advisor+":")
    text = text.replace('CUSTOMER', customer+":")
    return text


def save_to_file(originaltext, summary):
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, "summaries")
    counter = 0
    for file in os.listdir(directory):
        counter = counter + 1
    filename =  str(counter+1)+'.txt'
    file = open("summaries/"+filename, 'w', encoding='utf-8', errors='ignore')
    file.write("original:\r\n"+ originaltext+"\r\nSummary:\r\n"+summary)
    file.close()
    return filename