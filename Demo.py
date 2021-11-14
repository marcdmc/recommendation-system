import PySimpleGUI as sg
import random
import json
import requests
import numpy as np
import PIL
from PIL import Image
import io
import base64
import os
import pickle
from haversine import haversine
import math
from scipy.stats import norm

sg.theme('Reddit')

USER_LOC = (41.38308909469157, 2.115649028217146)

# Open data
test_f = open(".test_reviews.json")
train_f = open(".train_reviews.json")
test = json.load(test_f)
train = json.load(train_f)
data = test + train

# Load r from LDA
rs_file = open("r", 'rb')
r = pickle.load(rs_file)
print(r[0])

# Compute ratings
ratings = []
for restaurant in data:
    ratings.append(restaurant['rating'])

'''
def fun_score(rating, r_pos, u_pos, r, u):
    rating_score = norm.cdf(rating, loc=np.mean(ratings), scale=np.std(ratings))
    r[0] = 1/(1+haversine(r_pos, u_pos, unit='km'))
    if math.isnan(r[0]):
        print("NaN")
    #print(u, r)
    return rating_score*(np.dot(u, r))
    '''

def fun_score(rating, r_pos, u_pos, r, u):
  # The rating (5-star sistem) distribution is normal. We linealize the rating:
  rating_score = norm.cdf(rating, loc=np.mean(ratings), scale=np.std(ratings))

  # We want distance to be inversly proportional to restaurant score
  r[0] = 1/(1+haversine(r_pos, u_pos, unit='km'))
  if math.isnan(r[0]):
        print("NaN")
  r = np.array(r) / np.sqrt(np.sum(np.array(r)**2))
  return rating_score*(np.dot(u, r))

def update_u(u, beta, review_rating, r):
    print(u)
    print(review_rating)
    print(r)
    r = np.array(r) / np.sqrt(np.sum(np.array(r)**2))
    u = (1/beta)*np.asanyarray(u) + (2*norm.cdf(review_rating, loc=np.mean(ratings), scale=np.std(ratings)) - 1)*np.asarray(r) 
    return u / np.sqrt(np.sum(u**2))

def filter(price, mode, type_rest, max_dist, u_pos):
  # podem afegir si està obert o no
  restaurants = []
  for restaurant in data:
    try:
      dist = haversine((restaurant['gps_coordinates']['latitude'], restaurant['gps_coordinates']['longitude']), USER_LOC, unit='km')
      if restaurant['price'] in price and restaurant['type'] in type_rest and max_dist > dist and (mode[0]*restaurant['service_options']['dine_in'] or mode[1]*restaurant['service_options']['takeout'] or mode[2]*restaurant['service_options']['delivery']):
        restaurants.append(restaurant)
    except:
      pass
  print(len(restaurants))
  return restaurants

mode = [0, 1, 1]
S = filter(['$', '$$', '$$$'], mode, [restaurant['type'] for restaurant in data], 10, USER_LOC)

id_to_r = dict(zip([restaurant['place_id'] for restaurant in S], r))

# Arreglem aquest cas que no sé per què va malament
r[11][1] = 0.2
r[11][2] = 0.2
r[11][3] = 0.2
r[11][4] = 0.2
r[11][5] = 0.2

# r es el vector de vectors de rs (probs) per cada restaurant
u = [1, 0, 0, 0, 0, 0]

#global u
valued_restaurants_u = [(restaurant['place_id'], fun_score(restaurant['rating'], (restaurant['gps_coordinates']['latitude'], restaurant['gps_coordinates']['longitude']), USER_LOC, id_to_r[restaurant['place_id']], u)) for restaurant in S]
valued_restaurants_u.sort(key=lambda x: x[1], reverse=True)
chosen_restaurants = valued_restaurants_u[0:max(int(0.15*len(valued_restaurants_u)), min(10, len(valued_restaurants_u)))]
chosen_restaurants_distribution = []
sum_p2 = 0
for restaurant in chosen_restaurants:
    if restaurant[1] < 0:
        aux = 1 - abs(restaurant[1])
    else:
        aux = restaurant[1]
    sum_p2 += aux**2
    chosen_restaurants_distribution.append(aux**2)
for i in range(len(chosen_restaurants_distribution)):
    chosen_restaurants_distribution[i] = chosen_restaurants_distribution[i] * (1 / sum_p2)
if len(chosen_restaurants) < 5:
    # Imprimir tots els restaurants que es puguin
    print("No hi ha suficients restaurants a recomanar")
suggestion = np.random.choice(len(chosen_restaurants), size=5, replace=False, p=chosen_restaurants_distribution)
suggestions = [chosen_restaurants[index] for index in suggestion]
print(suggestions)

# Make list of top restaurant indices
# TODO: Fer mes eficient
top_restaurants = []
for suggestion in suggestions:
    i = 0
    for restaurant in data:
        if restaurant['place_id'] == suggestion[0]:
            top_restaurants.append(i)
            break
        i += 1

def recommendation_step(u, restaurant_selected, values):
  #global u
  valued_restaurants_u = [(restaurant['place_id'], fun_score(restaurant['rating'], (restaurant['gps_coordinates']['latitude'], restaurant['gps_coordinates']['longitude']), USER_LOC, id_to_r[restaurant['place_id']], u)) for restaurant in S]
  valued_restaurants_u.sort(key=lambda x: x[1], reverse=True)
  chosen_restaurants = valued_restaurants_u[0:max(int(0.15*len(valued_restaurants_u)), min(10, len(valued_restaurants_u)))]
  chosen_restaurants_distribution = []
  sum_p2 = 0
  for restaurant in chosen_restaurants:
    if restaurant[1] < 0:
      aux = 1 - abs(restaurant[1])
    else:
      aux = restaurant[1]
    sum_p2 += aux**2
    chosen_restaurants_distribution.append(aux**2)
  for i in range(len(chosen_restaurants_distribution)):
    chosen_restaurants_distribution[i] = chosen_restaurants_distribution[i] * (1 / sum_p2)
  if len(chosen_restaurants) < 5:
    # Imprimir tots els restaurants que es puguin
    print("No hi ha suficients restaurants a recomanar")
  suggestion = np.random.choice(len(chosen_restaurants), size=5, replace=False, p=chosen_restaurants_distribution)
  suggestion = [chosen_restaurants[index] for index in suggestion]
  print(suggestion)

  #restaurant_selected = int(input('Select restaurant from 1 to 5:'))
  print('Selected ' + str(restaurant_selected))
  recommendation_score = float(values[8])
  #recomendation_score = float(input('Please, give us a valuation (1 to 5 stars) on the recomendation:'))
  u = update_u(u, 1.2, recommendation_score, id_to_r[chosen_restaurants[restaurant_selected - 1][0]])
  return suggestion, u

#recommendation_step()

# Aquí fem tot el càlcul dels restaurants recomanats
# get_restaurant_scores() ---> ja et retorna la loss function de cada restaurant

'''
scores = [(scores[i],i) for i in range(len(scores))]
scores.sort()
'''

# Download restaurant thumbnails
def download_thumbnails(top_restaurants):
    for index in top_restaurants:
        restaurant = data[index]
        image_url = restaurant['thumbnail']
        try:
            img_data = requests.get(image_url).content
            with open('./'+ restaurant['title'] +'.png', 'wb') as handler:
                handler.write(img_data)
        except:
            print('Image for ' + restaurant['title'] + 'not found!')

download_thumbnails(top_restaurants)

def get_image(path, resize=None):
# Prepare image for processing
    if isinstance(path, str):
        img = PIL.Image.open(path)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(path)))
        except Exception as e:
            dataBytesIO = io.BytesIO(path)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.ANTIALIAS)
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()

# Make list of restaurants
restaurants_list = []
restaurant_names = []
for index in top_restaurants:
    restaurant = data[index]

    name_punctuation = sg.Column(
        [[sg.Text(restaurant['title'], background_color='#a3c1ad', enable_events=True, font=(None, 12))],
        [sg.Text(str(restaurant['type']), background_color='#a3c1ad')]],
        background_color='#a3c1ad')
    image = sg.Text("", background_color='#a3c1ad')
    try:
        image = sg.Column([[sg.Image(get_image(restaurant['title']+'.png'), enable_events=True, size=(40,40))]],
                background_color='#a3c1ad', element_justification='right', justification='right', vertical_alignment='center',
                expand_x=True, expand_y=True)
    except:
        pass
    restaurant_element = [[name_punctuation, image]]
    print(len(restaurant_element))

    restaurants_list.append([sg.Column(restaurant_element, background_color='#a3c1ad', size=(480,60))])
    restaurant_names.append(restaurant['title'])

# Initialize chosen restaurant
text1 = sg.Text('', background_color='#a3c1ad', enable_events=True, font=(None, 12))
text2 = sg.Text('', background_color='#a3c1ad')
name_punctuation = sg.Column(
    [[text1], [text2]], background_color='#a3c1ad')
chosen_image = sg.Image(get_image('gamba.png'), enable_events=True, size=(40,40), expand_y=True, background_color='#a3c1ad')
image = sg.Column([[chosen_image]],
                background_color='#a3c1ad', element_justification='right', justification='right', vertical_alignment='center',
                expand_x=True, expand_y=True)
rest_chosen = sg.Column([[name_punctuation, image]], size=(380,60), visible=False, background_color='#a3c1ad')

# Filtering options panel
price = sg.Column([[sg.Text('Price')],[sg.Checkbox('$')],[sg.Checkbox('$$')],[sg.Checkbox('$$$')]])
mode = sg.Column([[sg.Text('Mode')],[sg.Checkbox('Mode')],[sg.Checkbox('dine_in')],[sg.Checkbox('take_out')],[sg.Checkbox('delivery')]])
#type_ = sg.Column([[sg.Checkbox('a')]])
max_dist = sg.Column([[sg.Text('Max distance (km)')],[sg.Input('10', size=10, enable_events=True)]])

filter_buttons = [[price, mode, max_dist]]

# Initialize weights info
w_text = ""
for i in range(len(u)):
    w_text += "u" + str(i) + " = " + str(u[i]) + "\n"
print(w_text)
user_text_weights = sg.Text(w_text)

opinion_bar = [sg.Input('4.5', size=50)]

chosen_restaurant = [sg.Frame('Previously chosen restaurant',[[rest_chosen]], size=(300,100))]
user_profiling = [sg.Frame('User weights',[[sg.Column([[user_text_weights]], size=(300,200), element_justification='center')]])]

listofrestaurants = [[sg.Column(restaurants_list, size=(500,330))]]

left_panel = sg.Frame('Restaurant list', listofrestaurants)
right_panel = sg.Column([chosen_restaurant,[sg.Text('Give a rating:')],opinion_bar,user_profiling])

filter_options = [sg.Frame('Options', [[sg.Column(filter_buttons, size=(830,100))]])]
main = [sg.Column([[left_panel,right_panel]])]

# Title
title = [sg.Column([[sg.Text('Recommendation system using Latent Dirichlet Allocation', font=(None,16))]], element_justification='center', justification='center')]

layout = [title, filter_options, main]

window = sg.Window('Recommender', layout, resizable=True)

while True:
    event, values = window.read()
    print(event, values)

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    if event in restaurant_names:
        rng = restaurant_names.index(event)
        for rest in top_restaurants:
            if event == data[rest]['title']:
                pos = rest
        restaurant = data[pos]

        rest_chosen.update(visible=True)
        name_punctuation.update(visible=True)
        text1.update(event)
        text2.update(restaurant['type'])
        try:
            chosen_image.update(get_image(restaurant['title']+'.png'), size=(40,40))
        except:
            chosen_image.update(visible=False)
        print(event)
        print(u)
        suggestions, new_u = recommendation_step(u, rng, values)
        print('Updated us ' + str(new_u))
        u = new_u

        w_text = ""
        for i in range(len(u)):
            w_text += "u" + str(i) + " = " + str(u[i]) + "\n"
        print(w_text)
        user_text_weights.update(w_text)

        print(top_restaurants)
        # Update top restaurants
        top_restaurants = []
        for suggestion in suggestions:
            i = 0
            for restaurant in data:
                if restaurant['place_id'] == suggestion[0]:
                    top_restaurants.append(i)
                    break
                i += 1

        print(top_restaurants)

        download_thumbnails(top_restaurants)

        restaurants_list = []
        restaurant_names = []
        i = 0
        for index in top_restaurants:
            restaurant = data[index]
            try:
                img = get_image(restaurant['title']+'.png')
            except:
                img = ''
            
            print(listofrestaurants[0][0].Rows[i][0].Rows[0])
            # Update text and image fields
            listofrestaurants[0][0].Rows[i][0].Rows[0][0].Rows[0][0].update(restaurant['title'])
            listofrestaurants[0][0].Rows[i][0].Rows[0][0].Rows[1][0].update(str(restaurant['type']))
            listofrestaurants[0][0].Rows[i][0].Rows[0][1].Rows[0][0].update(get_image(restaurant['title']+'.png'), size=(40,40))
            i += 1

            #restaurants_list.append([sg.Column(restaurant_element, background_color='#a3c1ad', size=(480,60))])
            restaurant_names.append(restaurant['title'])

        # Llegir el restaurant que s'escull i després tornar a fer update dels pesos
        # guardar millor restaurant d'abans
        # get_scores()
        # update_weights(chosen_restaurant)

window.close()


