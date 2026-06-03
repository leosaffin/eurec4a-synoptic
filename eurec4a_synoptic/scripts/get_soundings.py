import datetime

import requests

# URL generated from https://weather.uwyo.edu/upperair/sounding.shtml
# Replace e.g. 2020-01-20 12:00:00, with formated datetime
url = (
    "https://weather.uwyo.edu/wsgi/sounding?datetime={}&id=78954&src=BUFR&type=TEXT:CSV"
)
datetime_format = "%Y-%m-%d %H:%M:%S"


def main():
    time = datetime.datetime(2020, 1, 20, 12)
    end_time = datetime.datetime(2020, 2, 20, 12)
    dt = datetime.timedelta(days=1)

    save_as = "grantley_adams_sounding_{}.csv"
    save_as_datetime_format = "%Y%m%dT%H%M"
    while time <= end_time:
        result = requests.get(url.format(time.strftime(datetime_format)))
        filename = save_as.format(time.strftime(save_as_datetime_format))
        print(filename)
        with open(filename, "w") as f:
            f.write(result.text)

        time += dt


if __name__ == "__main__":
    main()
