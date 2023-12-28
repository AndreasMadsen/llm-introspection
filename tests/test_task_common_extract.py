
from introspect.tasks._common_extract import extract_list_content, extract_ability, extract_paragraph

def test_task_extract_ability():
    c = extract_ability

    # yes
    assert c('Yes') == 'yes'
    assert c('yes') == 'yes'
    assert c('Yes.') == 'yes'
    assert c('yes.') == 'yes'

    # no
    assert c('No') == 'no'
    assert c('no') == 'no'
    assert c('No.') == 'no'
    assert c('no.') == 'no'

    # failure
    assert c('positive') == None
    assert c('negative') == None

def test_task_extract_paragraph():
    c = extract_paragraph

    p = ('This film is reminiscent of Godard\'s Masculin, féminin, and I highly'
         ' recommend checking out both films. The film boasts two standout'
         ' elements - the natural acting and the striking photography. While some'
         ' may find certain aspects of the film silly, I believe it\'s a charming'
         ' and thought-provoking exploration of human relationships. Lena Nyman'
         ' delivers an endearing performance, bringing depth and nuance to her'
         ' character. Unlike Godard\'s work, which can sometimes feel overly'
         ' cerebral, this film offers a refreshing change of pace. It\'s a'
         ' delightful reflection of the era and culture that produced it.'
         ' Rating: 8/10.')
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Paragraph: {p}') == p
    assert c(f'Paragraph: "{p}"') == f'{p}'
    assert c(f'Paragraph:\n\n{p}') == p
    assert c(f'{p}') == p

    # Extra double new-line
    p = ('If the talented team behind "Zombie Chronicles" reads this, here\'s some feedback:\n'
         '\n'
         '1. Kudos for incorporating creative close-ups in the opening credits!'
         ' However, consider saving some surprises for later to enhance the twists.\n'
         '2. Your cast did an admirable job with the resources available. For'
         ' future projects, investing in acting workshops or casting experienced'
         ' actors could elevate the overall performance.\n'
         '3. The historical setting was a unique touch. To further enhance the'
         ' immersion, consider adding more period-specific details in set design'
         ' and costuming.')
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Paragraph: {p}') == p
    assert c(f'Paragraph: "{p}"') == f'{p}'
    assert c(f'Paragraph:\n\n{p}') == p
    assert c(f'{p}') == p

    # Contains quoutes
    p = ('I was very disappointed in [REDACTED]. I had been waiting a really long time'
         ' to see it and I finally got the chance when it re-aired Thursday night on'
         ' [REDACTED]. I love the first three "[REDACTED]" movies but this one was'
         ' nothing like I thought it was going to be. The whole movie was [REDACTED]'
         ' and depressing, there were way too many [REDACTED], and the editing was'
         ' very poor - too many scenes out of context. I also think the death of'
         ' [REDACTED] happened way too soon and [REDACTED]\'s appearance in the movie'
         ' just didn\'t seem to fit. It seemed like none of the actors really wanted'
         ' to be there - they were all lacking [REDACTED]. There seemed to be no'
         ' interaction between [REDACTED] and [REDACTED] at all.')
    assert c(f'Here is the redacted version of the paragraph:\n\n{p}') == p
    assert c(f'Here is the redacted version of the paragraph:\n\n"{p}"') == p
    assert c(f'Sure! Here is the redacted version of the paragraph:\n\n{p}') == p
    assert c(f'Sure! Here is the redacted version of the paragraph:\n\n"{p}"') == p
    assert c(f'Paragraph: {p}') == p
    assert c(f'Paragraph: "{p}"') == f'{p}'
    assert c(f'Paragraph:\n\n{p}') == p

    # Incorrect output
    assert c(f'Sure! Here is the redacted version of the paragraph:') == None
    assert c(f'Sure! Here is the redacted version of the paragraph:\n') == None
    assert c(f'Sure! Here is the redacted version of the paragraph:\n\n') == None
    assert c(f'Sure! Here is the redacted version of the paragraph:\n\n""') == None
    assert c(f'Here are some suggestions based on your request:\n\n<ul><li>ABC</li><li>ABC</li></ul>') == None
    assert c(f'Here are some suggestions based on your request:\n\n<ol><li>ABC</li><li>ABC</li></ol>') == None

    # HTML processing
    assert c((
        'Here is the edited version of the paragraph:\n'
        '\n'
        '<p>My girlfriend once brought around The Zombie Chronicles for us to watch as a joke:</p>\n'
        '<ul>\n'
        '  <li>Drinking bleach</li>\n'
        '  <li>Rubbing sand in my eyes</li>\n'
        '  <li>Writing a letter to Brad Sykes and Garrett Clancy</li>\n'
        '</ul>\n'
        '\n'
        '<p>Garrett Clancy, aka Sgt. Ben Draper, wrote this? The guy couldn\'t even dig a hole properly.</p>'
    )) == (
        'My girlfriend once brought around The Zombie Chronicles for us to watch as a joke:\n'
        '\n'
        '* Drinking bleach\n'
        '* Rubbing sand in my eyes\n'
        '* Writing a letter to Brad Sykes and Garrett Clancy\n'
        '\n'
        'Garrett Clancy, aka Sgt. Ben Draper, wrote this? The guy couldn\'t even dig a hole properly.'
    )

    p = (
        'Watched on August 3rd, 2003 - 2 out of 10(Director-Brad Sykes):'
        ' A mindless 3-D movie about flesh-eating zombies set in a three-story building.'
        ' Although it may seem exciting at first, the lack of plot development and repetitive'
        ' nature of the film make for a boring experience.')
    assert c((
        f'<blockquote>\n\n\n{p}</blockquote>'
    )) == p

    assert c((
        'To make the sentiment positive, we need to change some words and phrases. Here are the changes made:\n'
        '\n'
        '- Instead of "Not much to say", we changed it to "There isn\'t much to say."\n'
        '- We removed "A plot you can pretty much peg," because it sounds negative.\n'
        '- We replaced "Nothing overly wrong with this film," with "It wasn\'t bad."\n'
        '- We added "Overall, it was enjoyable." at the end of the sentence.\n'
        '\n'
        'Here\'s the updated paragraph:\n'
        '\n'
        f'{p}'
    )) == p

    assert c((
        'To make the sentiment positive, I suggest removing negative words or phrases such'
        ' as "overrated", "unforgivable", "dullness", "microwave popcorn", "spawned",'
        ' "remake", "random", "quirky", "shouldn\'t", and "assume". Instead, focus'
        ' on highlighting the good aspects of the film such as the performances,'
        ' iconic characters, and impact on the genre. Here\'s an example:\n'
        '\n'
        f'{p}'
    )) == p

    p = (
        'Good movie overall. It has great action sequences,'
        ' good acting performances from Eddie Murphy and Charles Dance, and some funny moments.'
        ' Although there may have been some flaws in certain aspects like special effects and'
        ' pacing, it still manages to entertain viewers.'
    )
    assert c((
        'Here\'s an example:\n'
        '\n'
        'Positive Sentiment:\n'
        '\n'
        '\n'
        '\n'
        f'{p}\n'
    )) == p

def test_task_extract_list_content():
    c = extract_list_content

    # ordered
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '1. Awful\n'
             '2. Worst') == ['Awful', 'Worst']

    # unordered *
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '* Awful\n'
             '* Worst') == ['Awful', 'Worst']

    # unordered -
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '- Awful\n'
             '- Worst') == ['Awful', 'Worst']

    # no space
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '*Awful\n'
             '*Worst') == ['Awful', 'Worst']

    # qouted
    assert c('Sure! Here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '* "fun" (appears twice)\n'
             '* "great" (appears three times)\n'
             '* "spectacular"\n'
             '* "superb"\n'
             '* "totally recommend"\n'
             '* "talent"\n') == ['fun', 'great', 'spectacular', 'superb', 'totally recommend', 'talent']

    # Fancy dots and tabs
    assert c('Sure! Here are the most important words for determining the sentiment of the given paragraph:\n'
             '\n'
             '•\tFugly\n'
             '•\tAnnoying\n'
             '•\tAwful\n') == ['Fugly', 'Annoying', 'Awful']

    #
    # Mistral (these outputs are most common with mistral)
    #
    # one line qouted strings
    assert c('The most important words for determining the sentiment of this paragraph are:'
              ' "like," "loathe," "crushing bore," and "bad."') == ['like', 'loathe', 'crushing bore', 'bad']
    assert c('The most important words for determining the sentiment of this paragraph are'
            ' "like," "loathe," "crushing bore," and "bad."') == ['like', 'loathe', 'crushing bore', 'bad']
    assert c('Determining the sentiment of the paragraph requires the presence of certain words such as'
            ' "fabulous," "sure-fire hit," "good time," and "worth watching."') == ['fabulous', 'sure-fire hit', 'good time', 'worth watching']
    assert c('"like," "loathe," "crushing bore," and "bad."') == ['like', 'loathe', 'crushing bore', 'bad']
    assert c('"like", "loathe", "crushing bore", and "bad".') == ['like', 'loathe', 'crushing bore', 'bad']

    # stange qoute
    assert c('*very minor spoiler*, *actor who played Carlito*, *Luis Guzman*.') == ['very minor spoiler', 'actor who played Carlito', 'Luis Guzman']

    # strange content
    assert c('The most important words for determining the sentiment of the paragraph are'
             ' "enjoyed," "great," "liked," "fun," and ":)".') == ['enjoyed', 'great', 'liked', 'fun', ':)']

    # unqouted
    assert c('like, loathe, "crushing bore," and bad.') == ['like', 'loathe', 'crushing bore', 'bad']
    assert c('Mistake, acted, sang, Oscars.') == ['Mistake', 'acted', 'sang', 'Oscars']
    assert c('Notable words: plot, events, director, dreams.') == ['plot', 'events', 'director', 'dreams']

    # multiple lines
    assert c('Positive: beautiful, drop-dead gorgeous, fun, good time\n'
             'Negative: brain-rotting, rancid, annoying, waste of money') == [
                'beautiful', 'drop-dead gorgeous', 'fun', 'good time',
                'brain-rotting', 'rancid', 'annoying', 'waste of money'
            ]
