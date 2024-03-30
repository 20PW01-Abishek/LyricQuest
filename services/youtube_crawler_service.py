class youtube_crawler_service:
    @staticmethod
    def GetYtVideo(query):
        import requests
        from bs4 import BeautifulSoup
        import json
        import re

        r = requests.get('https://www.youtube.com/results?search_query={0}'.format(query))
        soup = BeautifulSoup(r.content, 'html.parser')
        script_tags = soup.find_all('script')
        ytInitialData_json = None

        for script_tag in script_tags:
            if 'ytInitialData' in script_tag.text:
                match = re.search(r'ytInitialData\s*=\s*({.*?});', script_tag.text)
                if match:
                    ytInitialData_json = json.loads(match.group(1))
                    break

        data = ytInitialData_json.get('contents')
        id = data.get('twoColumnSearchResultsRenderer', {})\
                .get('primaryContents', {})\
                .get('sectionListRenderer', {})\
                .get('contents', [{}])[0]\
                .get('itemSectionRenderer', {})\
                .get('contents', [{}])[0]\
                .get('videoRenderer', {})\
                .get('videoId')

        song_url = f'https://www.youtube.com/watch?v={id}'
        hyperlink = f'<a href="{song_url}" target="_blank">Listen to the song now!</a>'
        return hyperlink