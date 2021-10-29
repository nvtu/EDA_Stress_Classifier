import os
import os.path as osp
import requests
import json


# To get access token, refer to the API documentation on https://dev.fitbit.com/build/reference/web-api/oauth2/
# It is a GET request and it is quite tricky to get the access token properly.
# 1) Go to https://dev.fit bit.com/apps --> Login
# 2) Change the Callback URI to your machine public ip which I would name $machine_public_ip
# 3) https://www.fitbit.com/oauth2/authorize?response_type=token&client_id=22BBJ4&redirect_uri=https%3A%2F%2F{$machine_public_ip} \
# &scope=activity%20nutrition%20heartrate%20location%20nutrition%20profile%20settings%20sleep%20social%20weight&expires_in=604800
# access_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMkJCSjQiLCJzdWIiOiI3WUhXUkciLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcmFjdCBybG9jIHJ3ZWkgcmhyIHJwcm8gcm51dCByc2xlIiwiZXhwIjoxNjA5Nzk1MjcxLCJpYXQiOjE1ODAyMzMzMjh9.nhsTvQ7OrClze6b5pYxN3T2KqDU2sO-XpxnlseukM8o'
access_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMkJCSjQiLCJzdWIiOiI3WUhXUkciLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcmFjdCBybG9jIHJ3ZWkgcmhyIHJwcm8gcm51dCByc2xlIiwiZXhwIjoxNjI4NjQzMjY1LCJpYXQiOjE2MjgwMzg1MjJ9.JAEDFtI8Vfpg5CPW7jvdvITEnLViBYhqSx9YZuxELgI'

def get_api_url(api_name):
    url = None
    if api_name == 'heart_rate':
        url = 'https://api.fitbit.com/1/user/-/activities/heart/date'
    return url


def get_heart_rate_intra(date, start_time=None, end_time=None):
    '''
    date: The date, in the format yyyy-MM-dd or today.
    start-time: The start of the period, in the format HH:mm. Optional.
    end-time: The end of the period, in the format HH:mm. Optional.
    '''
    api_name = 'heart_rate'
    api_url = get_api_url(api_name) + '/{}/1d/1sec'.format(date)
    if start_time != None:
        if end_time == None:
            raise ValueError('end_time parameter must be not null value!!!')
        api_url += '/time/{}/{}'.format(start_time, end_time)
    api_url += '.json'
    header = { 'Authorization' : 'Bearer {}'.format(access_token) }
    r = requests.get(api_url, headers=header)
    data = r.json()
    return data


if __name__ == '__main__':
    dates = ['2020-01-06', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-15', '2020-01-20', '2020-01-21', '2020-03-01', '2020-03-04']
    parent_dir = '/'.join(os.getcwd().split('/')[:-1])
    output_folder_path = osp.join(parent_dir, 'MyDataset', 'Fitbit', 'HeartRate')
    if not osp.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for date in dates:
        output_file_path = osp.join(output_folder_path, '%s.json' % date)
        data = get_heart_rate_intra(date)
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)