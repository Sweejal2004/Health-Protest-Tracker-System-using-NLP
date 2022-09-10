from django.urls import path
from .views import *






urlpatterns = [
    path('', home, name='home'),
    path('get-data/', get_data, name='get-data'),
    path('get-map/', make_thing, name='get-map'),
    path('get-country', get_country, name='get-country'),
    path('get-search', get_search, name='get-search'),
    path('get-card-data', get_card_data, name='get-card-data'),
    path('another-try/', another_try, name='another-try'),
    path('export', export_users_csv, name='export_users_csv'),
]
