import sys

from flask import Flask
from flask import render_template, request, redirect, url_for
import requests
import json

import BARTGermanDialogs
import dialog_data
import t5GermanNews
import preprocessing
from dialog_data import DialogData
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["textInput"]
        if 'translated' in request.form:
            translated = True
        else:
            translated = False
        #wordsplit = request.form["wordsplit"]
        advisor = request.form["advisor"]
        customer = request.form["customer"]
        return redirect(url_for("t5",
                                summarizedText=text,
                                translated=translated,
                                #wordsplit=wordsplit,
                                advisor=advisor,
                                customer=customer))
    else:
        return render_template('home.html', summarizedText="test")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/t5', methods=["GET", "POST"])
def t5():
    if request.method == "POST":
        return redirect(url_for("home"))
    else:
        model = t5GermanNews.prepareTrainedModel()
        translated = request.args.get('translated')
        summarized_text = request.args.get('summarizedText')
        advisor_name = request.args.get("advisor")
        customer_name = request.args.get("customer")
        #wordsplit = int(request.args.get('wordsplit'))

        text_data = DialogData(summarized_text)
        text_data.replace_names(advisor_name, customer_name)
        if translated == "False":
            text_data.translate()

        print("Initializing model...")
        model = BARTGermanDialogs.prepare_trained_model()
        summaries = []
        summary_parts = {}
        dialog_parts = text_data.get_elements()

        separate_summary = dialog_parts[0]
        for text_element in separate_summary:
            print(f"starting summarization with length {len(text_element)}...")
            summary = BARTGermanDialogs.summarize(model, text_element)
            summaries.append(summary)
        summary_parts["introduction"] = "".join(summaries.copy())

        summaries = []
        separate_summary = dialog_parts[1]
        for text_element in separate_summary:
            print(f"starting summarization with length {len(text_element)}...")
            summary = BARTGermanDialogs.summarize(model, text_element)
            summaries.append(summary)
        summary_parts["keydata"] = "".join(summaries.copy())

        summaries = []
        separate_summary = dialog_parts[2]
        for text_element in separate_summary:
            print(f"starting summarization with length {len(text_element)}...")
            summary = BARTGermanDialogs.summarize(model, text_element)
            summaries.append(summary)
        summary_parts["portfolio"] = "".join(summaries.copy())

        summaries = []
        separate_summary = dialog_parts[3]
        for text_element in separate_summary:
            print(f"starting summarization with length {len(text_element)}...")
            summary = BARTGermanDialogs.summarize(model, text_element)
            summaries.append(summary)
        summary_parts["summary"] = "".join(summaries.copy())

        for key in summary_parts.keys():
            text = summary_parts[key]
            summary_parts[key] = dialog_data.DialogData.translate_back(text)

        text_as_whole = summary_parts["introduction"]+"\n"+summary_parts["keydata"] + "\n" +summary_parts["portfolio"] + "\n" +summary_parts["summary"]
        filename=preprocessing.save_to_file(text_data.get_formated_text(), text_as_whole)

        #text_to_show = t5GermanNews.summarize(model, summarized_text)
        return render_template("t5.html", intro=summary_parts["introduction"],
            keydata=summary_parts["keydata"],
            portfolio=summary_parts["portfolio"],
            summary=summary_parts["summary"],
            filename=filename)




if __name__ == '__main__':
    app.run()
