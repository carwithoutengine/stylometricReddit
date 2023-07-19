import praw

reddit = praw.Reddit(
    client_id="QHv39t6vcJE1zbV6dcyuhg",
    client_secret="AILcqPXTPlVY-o2Q0lEwSCMUuytf3g",
    user_agent="Comment Extraction (by u/SoC_CommentPull)",
    username="SoC_CommentPull",
    password="np.zTq^nLM8G!FT"              #all these are for API access
)

user = reddit.redditor('YourFriendMaryGrace') #username
for comment in user.comments.new(limit=None):
    print(comment.body)