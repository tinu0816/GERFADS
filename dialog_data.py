import preprocessing
import requests
import sys

import json

class DialogData:
    def __init__(self, text):

        text = preprocessing.remove_lines(text)
        #text = preprocessing.replace_names(blabla)

        text = text+"{end}"

        intro_index = text.find('{introduction}')
        key_index = text.find('{keydata}')
        portfolio_index = text.find('{portfolio}')
        summary_index = text.find('{summary}')

        #key & portfolio must not be empty:
        if key_index == -1 or portfolio_index == -1:
            raise Exception("transcript malformated")


        self.intro=""
        if intro_index != -1:
            self.intro = text[intro_index:text.find('{', intro_index+1)] #take text until next occurence of '{'
        self.keydata = text[key_index:text.find('{', key_index+1)]
        self.portfolio = text[portfolio_index:text.find('{', portfolio_index+1)]
        self.summary = ""
        if summary_index != -1:
            self.summary = text[summary_index:text.find('{', summary_index+1)]

        self.intro = self.intro.replace('{introduction}', '')
        self.keydata = self.keydata.replace('{keydata}', '')
        self.portfolio = self.portfolio.replace('{portfolio}', '')
        self.summary = self.summary.replace('{summary}', '')

        self.intro = self.intro.replace('{end}', '')
        self.keydata = self.keydata.replace('{end}', '')
        self.portfolio = self.portfolio.replace('{end}', '')
        self.summary = self.summary.replace('{end}', '')

        self.wordsplit = 500

    def replace_names(self, advisor_name, customer_name):
        self.intro = preprocessing.replace_names(advisor_name, customer_name, self.intro)
        self.keydata = preprocessing.replace_names(advisor_name, customer_name, self.keydata)
        self.portfolio = preprocessing.replace_names(advisor_name, customer_name, self.portfolio)
        self.summary = preprocessing.replace_names(advisor_name, customer_name, self.summary)

    def translate(self):
        self.intro = self.__translate_text(self.intro)
        self.keydata = self.__translate_text(self.keydata)
        self.portfolio = self.__translate_text(self.portfolio)
        self.summary = self.__translate_text(self.summary)

    def return_nr_of_elements(self, text, wordsplit):
        return_list = []
        list_element = ""
        for word in text.split():
            list_element += " " + word
            if len(list_element.split()) >= wordsplit:
                cut_index = list_element.rindex(": ")
                return_list.append(list_element[:cut_index])
                list_element = list_element[cut_index:]
        return_list.append(list_element)
        return return_list

    def get_elements(self):
        return_list = []
        return_list.append(self.return_nr_of_elements(self.intro, 500))
        return_list.append(self.return_nr_of_elements(self.keydata, 400))
        return_list.append(self.return_nr_of_elements(self.portfolio, 200))
        return_list.append(self.return_nr_of_elements(self.summary, 200))
        return return_list

    def get_formated_text(self):
        formated_text = ""
        if self.intro != '':
            formated_text = formated_text+"{introduction}\n"+self.intro+"\n"
        if self.keydata != '':
            formated_text = formated_text+"{keydata}\n"+self.keydata+"\n"
        if self.portfolio != '':
            formated_text = formated_text+"{portfolio}\n"+self.portfolio+"\n"
        if self.summary != '':
            formated_text = formated_text+"{summary}\n"+self.summary+"\n"
        return formated_text

    def __translate_text(self, text):
        print("Starting translation...", file=sys.stderr)
        # translate:
        url = "https://api-free.deepl.com/v2/translate"
        params = {
            "auth_key": "daaac6a2-43a1-7d7a-cff6-34f897f20e05:fx",
        }
        data = {
            "text": text,
            "target_lang": "EN",
        }
        response = requests.post(url, params=params, data=data)
        data = json.loads(response.text)
        translated = data["translations"][0]["text"]
        return translated

    @staticmethod
    def translate_back(text):
        print("Starting translation...", file=sys.stderr)
        # translate:
        url = "https://api-free.deepl.com/v2/translate"
        params = {
            "auth_key": "daaac6a2-43a1-7d7a-cff6-34f897f20e05:fx",
        }
        data = {
            "text": text,
            "target_lang": "DE",
        }
        response = requests.post(url, params=params, data=data)
        data = json.loads(response.text)
        translated = data["translations"][0]["text"]
        return translated