import json
from django.shortcuts import render
from .models import *
from django.http import  JsonResponse
import json
from .serializers import ConvertSerializer
from datetime import datetime, timedelta  
from datetime import date
from .forms import MyForm
from django.http import HttpResponse
import csv
import os

def home(request):
    try:
        location = request.POST.get('location')
        ddate = request.POST.get('d_date')
        
        request.session['data']=''
        request.session['data_d']=''

        all_data=protest_data.objects.all().order_by('location')
        if location:
            request.session['data'] = str(location)
            all_data=protest_data.objects.filter(location__icontains=location)
            
        
        if ddate:
            request.session['data_d'] = ddate     
            all_data=protest_data.objects.filter(d_date=ddate)  
            
        context={

        'all_data':all_data,
        'form':MyForm()
        
        }
        return render(request, 'anotherTRY.html', context)
    except Exception as e:
        print(e)
        return JsonResponse({'error': str(e)})
def make_thing(request):
    all_data=protest_data.objects.all()
    context={
        'all_data':all_data
    }
    return render(request, 'another.html', context)

def get_data(request):
    if request.method == 'POST':      
        country_name=request.POST.get('country')
        all_data=protest_data.objects.all()
        convert=ConvertSerializer(all_data, many=True)
        return JsonResponse(convert.data, safe=False)



def get_country(request):
    if 'term' in request.GET:
        qs = protest_data.objects.filter(location__icontains=request.GET.get('term'))
        companys = list()
        for comp in qs:
            companys.append(comp.location)
        return JsonResponse(companys, safe=False)           
    return render(request, 'another.html')




from pytz import timezone
def get_search(request):
    try:
        location = request.POST.get('location')
        ddate = request.POST.get('d_date')
        
        request.session['data']=''
        request.session['data_d']=''
        check_var=''

        all_data=protest_data.objects.all().order_by('location')
        if location:
            request.session['data'] = str(location)
            all_data=protest_data.objects.filter(location__icontains=location)
            
        
        if ddate:
            request.session['data_d'] = ddate     
            all_data=protest_data.objects.filter(d_date=ddate)  
            

        if not all_data:
            check_var='NO_DATA'
            
            

        context={

        'all_data':all_data,
        'form':MyForm(),
        'check_var':check_var
        
        }
        
        
        return render(request, 'data_table.html',context)
    
    except Exception as e:
        print(e)
        return JsonResponse({'error': str(e)})
   

def get_card_data(request):
    try:
        location = request.POST.get('country')
        all_data=protest_data.objects.filter(location__icontains=location)
        convert=ConvertSerializer(all_data, many=True)
        return JsonResponse(convert.data, safe=False)
    except Exception as e:
        print(e)
        return JsonResponse({'error': str(e)})
   
def another_try(request):
    return render(request, 'anotherTRY.html')




def export_users_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="users.csv"'

    writer = csv.writer(response)
    writer.writerow(['code','location', 'd_date', 'type_of_action', 'involved', 'tweet_url', 'trigger_for_protest', 'size_of_protest', 'duration', 'sentimentl_analysis','hashtags','retweets_count','likes_count'])

    all_data =protest_data.objects.all()
    print("type",type(all_data))

    alldata=request.session.get('data')
    d_data=request.session.get('data_d')
    if alldata:
        users=protest_data.objects.filter(location__icontains=alldata).values_list('code','location', 'd_date', 'type_of_action', 'involved', 'tweet_url', 'trigger_for_protest', 'size_of_protest', 'duration', 'sentimentl_analysis','hashtags','retweets_count','likes_count')
    elif d_data:
        users=protest_data.objects.filter(d_date=d_data).values_list('code','location', 'd_date', 'type_of_action', 'involved', 'tweet_url', 'trigger_for_protest', 'size_of_protest', 'duration', 'sentimentl_analysis','hashtags','retweets_count','likes_count')
    else:
        users = protest_data.objects.all().values_list('code','location', 'd_date', 'type_of_action', 'involved', 'tweet_url', 'trigger_for_protest', 'size_of_protest', 'duration', 'sentimentl_analysis','hashtags','retweets_count','likes_count')
    
    for user in users:
        writer.writerow(user)

    return response    


import pandas as pd


if(os.path.exists('static/csv/HealthProtest_Data.csv')):

    df = pd.read_csv('static/csv/HealthProtest_Data.csv')

    for f in range(len(df)):
        
        ddate=df.loc[f]['Datetime']
        
        # since = df['Datetime'].iloc[0]
        since=datetime.strptime(str(ddate), '%Y-%m-%d %H:%M:%S+00:00').replace(tzinfo=None)
        # since=datetime.strptime(str(ddate), '%Y-%m-%d %H:%M:%S')
        print(since)
        dddate=since.strftime('%Y-%m-%d')
        print(dddate)
        
        code=df.loc[f]['Country_Code']
        location=df.loc[f]['Country']
        involved=df.loc[f]['name']
        tweet_url=df.loc[f]['TweetUrl']
        trigger_for_protest=df.loc[f]['clean_text']
        sentimentl_analysis=df.loc[f]['sentiment']
        hashtags=df.loc[f]['Hashtags']
        media_url=df.loc[f]['Media']
        retweets_count=df.loc[f]['Retweets Counts']
        likes_count=df.loc[f]['Like Counts']
        print(dddate,location)
        protest_data.objects.create(
            code=code,
            location=location,
            d_date=dddate,
            involved=involved,
            tweet_url=tweet_url,
            trigger_for_protest=trigger_for_protest,
            sentimentl_analysis=sentimentl_analysis,
            hashtags=hashtags,
            retweets_count=retweets_count,
            likes_count=likes_count,
            media_url=media_url,
        )
    os.remove('static/csv/HealthProtest_Data.csv')

else:
    print("File not exist")
     


