class youtube_crawler_service:
    @staticmethod
    def get_yt_link(query):
        search_url = f"https://www.youtube.com/results?search_query={'+'.join(query.split())}"
        hyperlink = f'<a href="{search_url}" target="_blank">Youtube Search Results</a>'
        return hyperlink

