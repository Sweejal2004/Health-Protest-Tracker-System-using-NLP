# Generated by Django 3.2.7 on 2021-12-26 05:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='protest_data',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('location', models.CharField(max_length=50)),
                ('d_date', models.DateField(auto_now_add=True)),
                ('type_of_action', models.CharField(max_length=50)),
                ('involved', models.CharField(max_length=50)),
                ('tweet_url', models.CharField(max_length=50)),
                ('trigger_for_protest', models.CharField(max_length=100)),
                ('size_of_protest', models.CharField(max_length=50)),
                ('duration', models.CharField(max_length=50)),
                ('sentimentl_analysis', models.CharField(max_length=50)),
                ('image_videos', models.FileField(upload_to='protest_images/')),
                ('outcomes', models.CharField(max_length=50)),
                ('hashtags', models.CharField(max_length=50)),
                ('retweets_count', models.IntegerField(default=0)),
                ('likes_count', models.IntegerField(default=0)),
            ],
        ),
    ]
