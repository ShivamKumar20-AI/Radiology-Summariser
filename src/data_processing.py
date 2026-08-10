import os # this tool is used to check folders exists
import xml.etree.ElementTree as ET # this tool is used to read XML files
import pandas as pd # save data and create tables

xml_folder = "data/raw/ecgen-radiology" # path of the raw xml files

reports = [] # list to store the extracted data from XML files

xml_files = [] # list to store the names of the xml files in folders

for filename in os.listdir(xml_folder): # loops through the folder lookiing for files ending in .xml and adding them to the list of filenames
    if filename.lower().endswith(".xml"):
        xml_files.append(filename)

print("Folder exists:", os.path.exists(xml_folder)) # to check if the folder is there and if the path is correct
print("Number of XML files:", len(xml_files)) # cheks the number of files in the folder

for number, filename in enumerate(xml_files, start=1): # this loops through every filename

    print( # checks progress of the loop and prints the number of files processed as its a big folder and takes a while to load.
        "Reading file",
        number,
        "of",
        len(xml_files),
        ":",
        filename,
        flush=True
    )

    file_path = os.path.join(xml_folder, filename) # creates the full path to the file by joining the folder path and the filename

    tree = ET.parse(file_path) # reads the content and stores it for easy access
    root = tree.getroot() # gets structure of the XML file and stores it for easy access

    findings = "" # variable to store the findings text from the XML file
    impression = "" # variable to store the impression text from the XML file

    for element in root.iter(): # loops through all elements

        label = element.attrib.get("Label", "").upper() # gets label value and makes it uppercase for consistency

        if label == "FINDINGS":
            findings = " ".join(element.itertext()).strip() #gets the text and strips any leading/trailing whitespace

        if label == "IMPRESSION":
            impression = " ".join(element.itertext()).strip() # gets the text and strips any leading/trailing whitespace

    reports.append({ # saving a report
        "FINDINGS": findings,
        "IMPRESSION": impression
    })

data = pd.DataFrame(reports) # creating a table

os.makedirs("data/interim", exist_ok=True) # creates the folder if it doesnt exist(i created the whole structure beforehand)

data.to_csv( # saves the table as a csv file as the .csv suggests
    "data/interim/reports_flat.csv",
    index=False
)

print("Data processing complete")
print("Reports saved:", len(data))