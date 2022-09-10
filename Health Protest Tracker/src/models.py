from django.db import models
from django.db.models.fields import DurationField




class protest_data(models.Model):
    code                      =    models.CharField(max_length=11, null=True,blank=True)
    location                  =    models.CharField(max_length=50)
    d_date                    =    models.DateField(null=True, blank=True)
    type_of_action            =    models.CharField(max_length=50,null=True,blank=True)
    involved                  =    models.CharField(max_length=50)
    tweet_url                 =    models.CharField(max_length=50)
    trigger_for_protest       =    models.CharField(max_length=100)
    size_of_protest           =    models.CharField(max_length=50,null=True,blank=True)
    duration                  =    models.CharField(max_length=50,null=True,blank=True)
    sentimentl_analysis       =    models.CharField(max_length=50)
    image_videos              =    models.FileField(upload_to='protest_images/',null=True,blank=True)
    media_url                 =    models.CharField(max_length=250,null=True,blank=True)
    hashtags                  =    models.CharField(max_length=50)
    retweets_count            =    models.IntegerField(default=0,null=True,blank=True)   
    likes_count               =    models.IntegerField(default=0,null=True,blank=True)

    def __str__(self):
        return self.location


    
