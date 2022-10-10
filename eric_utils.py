from lxml import etree
import pandas as pd
import html
import json
import re
import os


__author__ = "Jon Ball"
__version__ = "Summer 2022"


nsmap = {'eric': 'http://www.eric.ed.gov',
         'dc': 'http://purl.org/dc/elements/1.1/',
         'dcterms': 'http://purl.org/dc/terms/'}


def parse_eric_api(path_to_xml):
    
    with open(path_to_xml, "rb") as xmlfile:
        tree = etree.parse(xmlfile)
        
    list_of_docs = []
    
    for doc in tree.xpath("//doc"):
        doc_metadata = {"title":"", "author":[], "description":"", "subject":[], "year":0, "issn":None}
        
        for elem in doc.getchildren():
            name = elem.attrib["name"]
            if name:
                
                match name:
                    case "title":
                        doc_metadata["title"] = html.unescape(elem.text)
                        
                    case "author":
                        doc_metadata["author"] = [
                            html.unescape(subelem.text) for subelem in elem.getchildren()
                        ]
                    case "description":
                        doc_metadata["description"] = clean_string(elem.text)
                        
                    case "subject":
                        doc_metadata["subject"] = [
                            html.unescape(subelem.text) for subelem in elem.getchildren()
                        ]
                    case "publicationdateyear":
                        doc_metadata["year"] = int(0 if elem.text is None else elem.text)

                    case "issn":
                        doc_metadata["issn"] = elem.getchildren()[0].text
                        
        list_of_docs.append(doc_metadata)

    print("%s docs parsed." % (len(list_of_docs,)))
        
    return list_of_docs


def parse_eric_database(path_to_xml, item_type):
    
    with open(path_to_xml, "rb") as xmlfile:
        tree = etree.parse(xmlfile)
        
    list_of_docs = []
    
    for doc in tree.xpath("//metadata"): #for each item record, identify metadata
        if (doc.xpath("./dc:type[text()='%s']" % item_type, namespaces=nsmap) 
            #if the item is an article
            and doc.xpath("./dc:language[text()='English']", namespaces=nsmap) 
            #and the article is in English
           ):
            
            d = {"subject":[], "author":[], "description":"", "title":"", "year":0, "source":None}
            
            subject_keywords = [
                elem.text for elem in doc.xpath("./dc:subject", namespaces=nsmap) if elem.text
            ]
            author_list = [
                elem.text for elem in doc.xpath("./dc:creator[@scheme='personal author']", namespaces=nsmap)
            ]
            description = doc.xpath("./dc:description", namespaces=nsmap)[0].text
            description = clean_string(
                description if description else ""
            )
            title = doc.xpath("./dc:title", namespaces=nsmap)[0].text
            title = html.unescape(
                title if title else ""
            )
            year = doc.xpath("./dc:date", namespaces=nsmap)[0].text
            year = int(
                year[:4] if year else 0
            )
            source = doc.xpath("./dc:source", namespaces=nsmap)[0].text
            source = html.unescape(source) if source else None
            
            d["subject"], d["author"], d["description"], d["title"], d["year"], d["source"] = \
            subject_keywords, author_list, description, title, year, source
        
            list_of_docs.append(d)
    
    del tree
        
    return list_of_docs


def clean_string(s):
    clean = html.unescape(s)
    #remove the initials of staff members who transcribed abstracts (DA)
    clean = re.sub(r"\s\(.{2,50}\)$", "", clean)
    #remove bracketed recommendations of relevant literature
    clean = re.sub(r"\s\[.{2,200}\]$", "", clean)
    #remove backslash escape
    clean = re.sub(r"\\", "", clean)
    return clean


def make_df(list_of_docs):
    df = pd.DataFrame.from_records(list_of_docs)
    df = df[(1990 <= df.year) &
            (df.year <= 2019)].reset_index()
    df["etm_input"] = df.apply(create_etm_inputs, axis=1)
    print("%s relevant articles published from 1990-2019." % df.shape[0])
    return df


def create_etm_inputs(df):
    return ". ".join([re.sub(r"\.$", "", df["title"]), df["description"]])


def save_author_list(df, name):
    authors = [author for doc in df["author"].tolist() for author in doc if author]
    authors = list(set(authors))
    print("%s individual authors indexed in %s publications from 1990-2019." % (len(authors), name))
    
    filename = name + "_authors.json"
    with open(os.path.join("eric_data", filename), "w") as outfile:
        json.dump(authors, outfile)
    print("Author list saved as %s" % (filename,))