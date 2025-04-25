from importlib import reload
import os
import sys
reload(sys) #some useful things which help the system run os and all #reload refreshes pythons internal settings
from flask import Flask, render_template, request, redirect, send_from_directory, make_response
#flask backend mai use hota bro
from werkzeug.utils import secure_filename
import imgtotxt #ye hai hamare custom images
import txttoimg
app = Flask(__name__)
s = dict()  # like 101:'a' 102:'b'
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', u="Upload Image", c="COMPRESS!", ul='/compressed', l="Upload Text")
#used to show an html file #request is to get data from the user
@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file(): #when u will upload file ja b user file upload kare to ye work karta hain.
    if request.method == 'POST':
        if 'fileToUpload' not in request.files:
            return 'No file part'
        f = request.files['fileToUpload']
        if f.filename == '':
            return 'No selected file'
        imgname = secure_filename(f.filename)
        f.save(imgname)
        imgtotxt.imgtotxt(imgname) #now image to text convert ho jaata hai jo input image was send okay
        return render_template('index.html', u="Image Uploaded!", c="COMPRESS!", ul='/compressed', l="Upload Text")
    #so return render is saying show the html page
@app.route('/compressed', methods=['GET', 'POST'])
def compress(): # post request jab submit karoge tab hi load karo
    if request.method == 'POST':
        os.system('gcc huffman.c -o huffman.exe') #this compiles your c file and makes a dot exe file.
        stream = os.popen('huffman.exe')
        output = stream.read().strip().split('\n') #this reads exe file and whatever it is seeing it is going to print
        #it like something like a stream of data is coming
        #the huffman.exe output is something like this number: its converted binary form from the c program
        keys = [int(i.split(': ')[0]) for i in output]
        data = [i.split(': ')[1] for i in output] #this is the string data character which we are storing in the list
        global s
        s = dict(zip(keys, data)) #dictionary like key:data
        dtbw = [] #means data to be written
        with open('test1.txt') as f:  #this file contains raw numbers jo stream mai print hue the
            for i in f:
                try:
                    dtbw.append(s[int(i)])
                except:
                    dtbw.append(i)
        dtbw = [str(i).strip() + '\n' for i in dtbw]
        with open('compressed.txt', 'w') as f:
            f.writelines(dtbw)
        #so now we have a file which makes compressed.txt from the initial file which was converted to txt from jpg
        response = make_response(send_from_directory('.', 'compressed.txt'))  #sends a response now the user can download the compressed data
        response.headers["Content-Disposition"] = "attachment; filename=compressed.txt"
        # Optional: Clean up temporary file
        # os.remove('compressed.txt')
        return response
    return "Invalid request method."
@app.route('/decomupload', methods=['GET', 'POST']) # from this route u will be able to decompress the image
def decom():
    if request.method == 'POST': #get post method if it is post method
        if 'txtToUpload' not in request.files:
            return 'No file part'
        f = request.files['txtToUpload']
        if f.filename == '':
            return 'No selected file'
        txtname = secure_filename(f.filename) #that compressed file is read as txtname
        f.save(txtname)
        return render_template('index.html', u="Image Uploaded!", c="COMPRESS!", ul='/compressed', l="Text Uploaded!")
    return redirect('/')
@app.route('/decompressed', methods=['GET', 'POST'])
def decompress(): #this method will search for a value:key pair instead of key:value pair like initially 123 ka 00101 to
    #ye 00101 ka 123 dhundega
    if request.method == 'POST':
        global s
        d = []
        with open('compressed.txt', 'r') as f:
            lines = f.read().splitlines()
        if not lines:
            return 'Compressed file is empty.'
        decoded_lines = lines[:-1]
        last_line = lines[-1] if lines else ""
        for val in decoded_lines:
            for key, value in s.items():
                if value == val:
                    d.append(key)
                    break
        # list d will have things like [123,34,45,56] like jo pehle initially tha
        d = [str(i) + '\n' for i in d]
        d.append(str(last_line))
        with open('decompressed.txt', 'w') as f:
            f.writelines(d)
        txttoimg.txttoimg('decompressed.txt')  #that list is converted to a file called this wo numbers usme jaayenge then image is made
        # response sent , ready for download
        response = make_response(send_from_directory('.', 'test.jpg'))
        response.headers["Content-Disposition"] = "attachment; filename=your_image.jpg"
        return response #response is written once the jpg image is ready
    return "Invalid request method."
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
#please note that this is a loseless compression algorithm
