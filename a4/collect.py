from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI

consumer_key = 'j9Cb7HC1AqFDr5r22IJ84DStL'
consumer_secret = 'tjGTkEcyQL29SSMcJs04kLIZR0PmyidZahCGpwO1eImMNWng3F'
access_token = '351402068-X6kgw36kCdQEOKQwCHqtLC2nRMVy46akMbGwy0xf'
access_token_secret = 'xPzA3tRdERTnqfd9EdTB1k7IUmjZeWeZa77ElNhbE9sc9'

# This method is done for you. Make sure to put your credentials in the file twitter.cfg.

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    ###TODO
    result = []
    f = open(filename)
    for line in f:
        result.append(line.rstrip('\n'))
    return result


# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO
    return robust_request(twitter, 'users/lookup', {'screen_name': screen_names})


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids

    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    result = robust_request(twitter, 'friends/ids', {'screen_name': screen_name, 'count': 5000})
    return sorted(result)


def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    for i in users:
        i['friends'] = get_friends(twitter,i['screen_name'])
    

def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    ###TODO
    result = []
    for i in users:
        result = result + i['friends']
    return Counter(result)

def output_user_and_friendsid(users):
	text_file = open("output_user.txt", "w")
	for i in sorted(users, key=lambda x: x['screen_name']):
		for j in i['friends']:
			text_file.write('%s\t%d\n' % (i['screen_name'], j))
	text_file.close()

def output_message(searchword):
	text_file = open("output_message.txt", "w")
	twitter = get_twitter()
	screen_name = [searchword]
	tweets = []
	for i in robust_request(twitter, 'search/tweets', {'q': screen_name, 'lang': 'en', 'count': 100}):
		tweets.append(i)
	# print('number of tweets: %d' % len(tweets))
	for i in range(len(tweets)):
		# print(str(tweets[i]['text']))
		text_file.write('%s\n' % str(tweets[i]['text']))
	text_file.close()


def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    add_all_friends(twitter, users)
    friend_counts = count_friends(users)
    output_user_and_friendsid(users)
    output_message('Warcraft')
    


if __name__ == '__main__':
    main()

# That's it for now! This should give you an introduction to some of the data we'll study in this course.
