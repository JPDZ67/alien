import requests
import lxml.html as lh
import pandas as pd


class Scrapper:
    def __init__(self):
        self.base_url = 'http://www.nuforc.org/webreports/'
        self.reports_url = 'http://www.nuforc.org/webreports/ndxevent.html'
        self.sightings = []

    def run(self):
        html = self.extract_html(self.reports_url)
        links = self.extract_links(html)
        self.get_sightings(links)
        return self.sightings_df()

    def extract_html(self, url):
        page = requests.get(url)
        return lh.fromstring(page.content)

    def extract_links(self, html):
        report_links = html.xpath('//tr')
        return self.find_links(report_links)

    def find_links(self, page_els):
        a_els = []
        for item in page_els:
            a_els.append(item.xpath('//a'))

        hrefs = []
        for a in a_els[0]:
            hrefs.append(a.get('href'))

        return hrefs

    def get_sightings(self, links):
        for link in links:
            page = self.extract_html(f"{self.base_url}{link}")
            rows = page.xpath('//tr')

            for row in range(1, len(rows)):
                report = []
                data = rows[row]

                for sighting in data.iterchildren():
                    text = sighting.text_content()
                    report.append(text)

                self.sightings.append(report)

    def sightings_df(self):
        df = pd.DataFrame(self.sightings,
                            columns=['Datetime', 'City', 'State', 'Shape', 'Duration', 'Summary', 'Posted'])
        df.to_csv('../raw_data/new_sightings.csv')
        return df
