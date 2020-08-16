# Text-Clustering-API
Implementation of a text clustering algorithm using Kmeans clustering in order to derive quick insights from unstructured text.
Please check the below links for details - 

+ https://machinelearningblogs.com/2017/01/26/text-clustering-get-quick-insights-from-unstructured-data/
+ https://machinelearningblogs.com/2017/06/23/text-clustering-get-quick-insights-unstructured-data-2/

## Docker Setup
0. Install [Docker](https://docs.docker.com/engine/installation/)
1. Run `git clone https://github.com/vivekkalyanarangan30/Text-Clustering-API`
2. Open docker terminal and navigate to `/path/to/Text-Clustering-API`
3. Run `docker build -t clustering-api .`
4. Run `docker run -p 8180:8180 clustering-api`
5. Access http://192.168.99.100:8180/apidocs/index.html from your browser [assuming you are on windows and docker-machine has that IP. Otherwise just use localhost]

## Native Setup
1. Anaconda distribution of python 2.7
2. `pip install -r requirements.txt`
3. Some dependencies from *nltk* (`nltk.download()` from python console and download averaged perceptron tagger)

### Run it
1. Place the script in any folder
2. Open command prompt and navigate to that folder
3. Type "python CLAAS.py"and hit enter
4. Go over to http://localhost:8180/apidocs/index.html in your browser (preferably Chrome) and start using.
