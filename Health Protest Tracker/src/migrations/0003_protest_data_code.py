# Generated by Django 3.2.7 on 2021-12-30 11:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('src', '0002_remove_protest_data_outcomes'),
    ]

    operations = [
        migrations.AddField(
            model_name='protest_data',
            name='code',
            field=models.CharField(blank=True, max_length=11, null=True),
        ),
    ]
