import random


class Playlist:
    def __init__(self):
        self.songs = []

    def add_song(self, title, artist):
        self.songs.append({"title": title, "artist": artist})

    def shuffle(self):
        random.shuffle(self.songs)


if __name__ == "__main__":
    playlist = Playlist()
    playlist.add_song("Song One", "Artist A")
    playlist.add_song("Song Two", "Artist B")
    playlist.add_song("Song Three", "Artist C")
    playlist.add_song("Song Four", "Artist D")
    playlist.add_song("Song Five", "Artist E")
    playlist.shuffle()
    for song in playlist.songs:
        print(f"{song['title']} by {song['artist']}")
