from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        
        company = request.form['brand']
        type = request.form['type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])

        
        str_touch = request.form['touchscreen']
        if str_touch == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0
        # ips = request.form['ips']
        str_ips = request.form['ips']
        if str_ips == 'Yes':
            ips = 1
        else:
            ips = 0

        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']
        
        # Calculate ppi
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
        
        
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
        query = query.reshape(1, 12)
        
        # Make the prediction
        predicted_price = int(np.exp(pipe.predict(query)[0]))
        
        
        return render_template('result.html', price=predicted_price)
    
    
    return render_template('home.html', brands=df['Company'].unique(), laptop_types=df['TypeName'].unique(),
                           resolutions=['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                        '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
                            ram_sizes = [2,4,6,8,12,16,24,32,64], ssd_sizes = [0,8,128,256,512,1024],
                           cpus=df['Cpu brand'].unique(), gpus=df['Gpu brand'].unique(), operating_systems=df['os'].unique(),hdd_sizes = [0,128,256,512,1024,2048])

if __name__ == '__main__':
    app.run(debug=True)
