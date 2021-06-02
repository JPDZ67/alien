import requests
import lxml.html as lh
import pandas as pd


class Scrapper:
    def __init__(self):
        self.base_url = 'http://www.nuforc.org/webreports/'
        self.reports_url = 'http://www.nuforc.org/webreports/ndxevent.html'
        self.sightings = []

    def run(self):
        """Returns a df with the scrapped data from the website and saves it into a csv"""
        html = self._extract_html(self.reports_url)
        links = self._extract_links(html)
        self.get_sightings(links)
        return self.sightings_df()

    def _extract_html(self, url):
        """Internal - Extracts html document from a url"""
        page = requests.get(url)
        return lh.fromstring(page.content)

    def _extract_links(self, html):
        """Internal - Extracts source href from html a tags"""
        report_links = html.xpath('//tr')
        return self._find_links(report_links)

    def _find_links(self, page_els):
        """Internal - Finds a (link) elements from html document"""
        a_els = []
        for item in page_els:
            a_els.append(item.xpath('//a'))

        hrefs = []
        for a in a_els[0]:
            hrefs.append(a.get('href'))

        return hrefs

    def get_sightings(self, links):
        """Internal - Navigates to report pages and extracts sightings content"""
        for link in links:
            print(f"Extracting data from - {link}")
            page = self._extract_html(f"{self.base_url}{link}")
            rows = page.xpath('//tr')

            for row in range(1, len(rows)):
                report = []
                data = rows[row]

                for sighting in data.iterchildren():
                    text = sighting.text_content()
                    report.append(text)

                self.sightings.append(report)

    def sightings_df(self):
        """Internal - Converts sightings to dataframe and saves them as csv"""
        df = pd.DataFrame(self.sightings,
                          columns=['datetime', 'city', 'state', 'shape', 'duration (seconds)', 'comments', 'date posted'])
        df.to_csv('../raw_data/new_sightings.csv')
        return df
