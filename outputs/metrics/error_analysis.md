# Error Analysis

Total predictions: 720
Total errors: 100

## Error rate by domain

| domain | total | errors | error_rate |
| --- | --- | --- | --- |
| hotel | 238 | 44 | 0.1849 |
| product | 482 | 56 | 0.1162 |

## Error rate by review length

| length_bucket | total | errors | error_rate |
| --- | --- | --- | --- |
| <50 | 288 | 46 | 0.1597 |
| 50-100 | 170 | 21 | 0.1235 |
| 100-200 | 167 | 27 | 0.1617 |
| >200 | 95 | 6 | 0.0632 |

## Five most confident wrong predictions by confusion pair

### Genuine Human -> Fake Human-Written

| id | review | predicted_probability |
| --- | --- | --- |
| 822 | the sheraton was a wonderful hotel! when me and my mom flew in we were really tired so we decided to take a quick nap. we did not want to get up! the beds are absolutely to die for. i wanted to take it home with me. the service was great and this was probably one of the biggest if not the biggest hotel i have ever stayed in. they had a really nice restaurant inside with excellent food. | 0.9524775743484497 |
| 2389 | i stayed at the hyatt for a two nights on a business trip. my room was oddly shaped to give a stunning view of the river and nbc building. both the room and the bathroom were spotless and the room had an air purifier which ran silently. the hotel was well staffed with individuals who were knowledgeable about both the hotel and the surrounding area. i would not hesitate to stay here again and plan to return with my wife for a weekend. | 0.920888364315033 |
| 2511 | i checked in at the affinia hotel in chicago shortly after dinner on tuesday, i was greeted by very frendly staff from the valet service to front desk. you can tell people were happy at the hotel. i was pleasantly surprised when i got to my room. the room was very well decorated and well thought out. the furniture was in the right location and there was lots of space. plus it was convenient with lots of power outlets the bedroom and washroom was very clean and i had a great sleep in their very comfortable bed. i was told by front desk i could choose from 6 different pillows, although i did not take advantage of this offer i thought what a great idea. once in a while you come across hotels with great service, well thought out rooms in a perfect location right off magnificent mile. when visiting chicago again i will definitely try and stay at affinia again | 0.9153891801834106 |
| 4169 | my family and i have just had a two week holiday in chicago, and we stayed for a week at the hilton towers. we had a fantastic time and enjoyed every minute of our stay at this incredible hotel. the sheer size of the place is breathtaking and the atmosphere is very friendly and hospitable. if you prefer discreet modern boutiqueytype hotels, maybe this would not be your thing. however if you enjoy old style glamour and glitz, you will be bowled over by this hotel. the place literally sparkles thanks to the enormous chandeliers that are everywhere in the lobby. tons of celebs have stayed here over the decades and the history of the hotel and past guests etc is fascinating. we got a very reasonable rate for our stay and could not fault anything about this very special place! | 0.9153000712394714 |
| 3070 | i just returned from a stay at this fabulous hotel what a treat! the hotel website offers a great rate, and what i found was affordable luxury. i asked for a high floor found myself with a fabulous view, in a room where there was an orchid spray in the bathroom, terrycloth robes and slippers, turndown service, roger gallet toiletries, a fluffy down conforter, and a wonderful pillowtop bed. the bathroom was so clean that it sparkled. the service was impeccable as well. even my room service breakfast was way above average with an incredibly generous basket of excellent croissants, brioches, pain au chocolat. the decor was unique the location convenient. check out the stunning flowers throughout the lobby. this is a topnotch hotel. | 0.8775602579116821 |

### Genuine Human -> AI-Generated

| id | review | predicted_probability |
| --- | --- | --- |
| 2330 | i really loved this really great story. i could not put the book down till i finished reading the whole wonderful story. thank you so very much for writing this wonderful story. | 0.9865812063217163 |
| 1810 | it is a very nice story. it is funny and loving . sometime i get mad at men who are so stubborn , but i guess we women are that way sometime to. it is romance and that is what the book is about and it is good. | 0.9603918790817261 |
| 2430 | i am very satisfied with the arcenciel water resistant bicycle bag. i am so satisfied i ordered a second one. | 0.9499837160110474 |
| 3898 | this show is getting worse and worse by the episode. | 0.9144439697265625 |
| 4044 | my children enjoyed this game when they were in the 2 to 4 year old range. there is a little memory practice going on combined with color matching. to play the game a bunch of cardboard fish are placed color side down in the center of the play area. each player is given a plastic boat with 4 slots in it. players use a plastic fishing pole with a suction cup on the end to pick up fish from the center. if the fish matches their boat's color the player gets to put the fish in his or her boat. if it does not match the color then the fish goes back into the center. this is where the memory aspect comes in. the other players need to remember where a fish was placed back in the center if it happened to be of their color. it does not take much memory to remember one fish location until your turn comes around, but this is really for the very youngest game players. the first player to fill their boat wins. there is a variant that let us it use the numbers on the colored fish to play the classic go fish game of making sets and asking other players for matches. this variant stretches the age appropriateness up to maybe 4 years old. the go fish game lost its appeal for my children once the youngest was 4. they moved on to more advanced memory games like the magic labyrinth board game and hide and eeek after that. the pieces are colorful and attractive. the game lasted fine through a couple of years of use. the suction cup worked okay for us. the game would probably work fine in a preschool classroom. all in all it is a fun little game for the very youngest of gamers at a cheap price. | 0.8594452738761902 |

### Fake Human-Written -> Genuine Human

| id | review | predicted_probability |
| --- | --- | --- |
| 3567 | pleasant staff and housekeeping. above average breakfast. the rooms were clean and the staff made sure we had fresh linnens. it was bedbug free!! i would definitely stay there again. | 0.9738775491714478 |
| 2310 | we decided to stay at the james last weekend with our kids. we reserved the 'family adventurer with paul frank'. it is a special offer by the hotel to entice visitors to see the local museums and aquarium. we ended up receiving 2 chicago city passes, which ultimately paid for themselves and 4 tickets for the chicago children's museum. my kids had a ball. we ended the day with a milk and cookies turn down service and a special pair of pajamas that my daughter adored. all in all, a great weekend where my kids were the focus. the adults certainly were not forgotten since we had a great breakfast in the room the next morning. | 0.9148111939430237 |
| 2820 | i received the type of room that i had reserved. cleanliness seems to be an issue with the maid staff. the carpet had debris and the bathroom needed attention. room service was adequate but not great. the walls seem to be paper then as i heard one neighbor's tv for most of the night and the other neighbor's late enjoyment. | 0.8972684741020203 |
| 3461 | on our first trip to chicago, my fiance and i stayed 2 nights on our anniversary and were pleasantly surprised by the hotel and the service we received from the personnel. in the past from other 'luxury hotels' we have paid much more and received much less. here the room were very spacious, comfortable and sleek. we spent come time in the hotel bar, le bar, and enjoyed a few of their specialty martinis that were second to none! we also enjoyed dinner at the hotel's cafe des architectes and received service some of the best service in quite some time. if traveling with a lover we recommend the 'air of romance' package. it was the best! 6 stars! | 0.8636358976364136 |
| 687 | this hotel is beautiful, it is very modern and has a very luxuary feel to it. the location is wonderful,it is close to magnificent mile and the navy pier, which was a great place to show the children on our vacation. we had the option of private parking which made a feel secure with our car,and the valet was very nice and helpful. i also take my work with me everywhere i go,so the wifi at the hotel was aplus. they have a wide selection of packages that can include anything from the couple that wants to get away or a family outing. this place is great for the price, due to the looks of the hotel you will be surprised at how affordable it is. this is very nice place and it gives you a feeling of being the queen or king for the duration of your stay. | 0.8432052731513977 |

### Fake Human-Written -> AI-Generated

| id | review | predicted_probability |
| --- | --- | --- |
| 738 | excellent staff and customer service, very clean and spotless. elegant and luxurious with a beautiful ocean view. the bed is very comfortable and relaxing. i give it a five star. | 0.5314033627510071 |

### AI-Generated -> Genuine Human

| id | review | predicted_probability |
| --- | --- | --- |
| 3347 | quiet, good airflow. love it. works great, just as advertised. | 0.9524240493774414 |
| 1336 | bring your hankies. fabulous. not a bad movie, just not a lot of action. | 0.9302384257316589 |
| 1119 | batteries shipped on time when they are delivered wish i could find one for | 0.8876434564590454 |
| 135 | nice product, increases wifi range. works well with my router. also, it is been a couple months since i | 0.8784306049346924 |
| 2463 | wow anything with a male lead, with a female lead, and a female lead. cannot wait for the next book in this series. great writing. i cannot wait to read the next in the series. i absolutely loved this book! i was given this book by a friend and i was immediately hooked. i cannot wait to read the next book in this series. | 0.8575244545936584 |
