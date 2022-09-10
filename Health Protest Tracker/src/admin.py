from django.contrib import admin
from .models import *



# Register your models here.


@admin.register(protest_data)
class ProtestAdmin(admin.ModelAdmin):
    list_display=('code','location','type_of_action','involved','tweet_url','trigger_for_protest','size_of_protest','duration','sentimentl_analysis','image_videos','hashtags','retweets_count','likes_count')