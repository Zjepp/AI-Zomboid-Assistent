# Project Zomboid AI Survival Assistant

## Project Zomboid

Project Zomboid is a brutal zombie survival game situated in "Knox Country", Kentucky USA. The goal is obviously surviving the undead, finding food and water before unevitably dying from disease, bites,... The game uses a "fake 3D system" with 2D floors, walls and terrain, the only 3D models in the game are currently the player, the zombies, animals and the cars.

## Problem Statement

Surviving in a harsh world like this where certain death lies around every corner waiting for a player not paying enough attention or being distracted by something else in the game. In this scenario players will need help in every way and I thought it would be really cool to make a model that could analyze the screen and decide what dangers are present and where they are. Like that the AI can help the player stay alive for longer and avoid making mistakes.

## Data

The biggest challenge in this project is surely the data, as mush as I looked I could not find any annotated data for this game. So I decided to dig through Steam community pictures, PZ wiki and many more online images. After I asked the wise worldwide web what "the best" annotation tools for computer vision where and landed at Roboflow. Using the free tier I started drawing bounding boxes around objects that seemed important for my model.

### Classes

The classes that came to mind when watching all these pictures were:

- Player
  This is the most important part, the model should find the player and see where danger is relative to the Player.
- Player Name
  Until this day I still do not really understand why i chose for Player Name, because in most pictures it is not even present. In the later models this class will be dropped.
- Zombie
  The biggest direct threat in the game are the zombies, player like creatures infected by the "Knox Virus" that can walk, run and sit.
- Dead zombie
  The more corpses on the floor the more chance of getting "corpse sickness" so avoid staying too long around them, but they are much safer than the alive undead.
- Window
  If the model can find windows in a screenshot then it can count that as a way out or escape from a dangerous situation
- Door
  Same as window a door can be way to freedom and safety (or the complete opposite)
- Tree
  Trees means the player is outside and surely has great vision of everything around them, but no trees block vision and after every tree or even the leaves because it is a 2D game can be zombie hidden
- Fire
  Fire is the second biggest direct threat to the safety of the Players health, but it does not occur that often in the game so this class is highly underrepresented in my dataset
- Campfire
  I thought it would be nice to differentiate between safe contained fire and big bad fire that can burn entire towns. This Class would also get dropped pretty fast, also because i had i think in total 3 screenshots of a campfire and to be honest a bad kept campfire can still burn down a forest so I merged it just with fire.

I ended up with 7 real classes: Door, Fire, Player, Tree, Zombie, Zombie Dead, window. Doors and windows were kept as single classes rather than splitting them into open/closed/broken states, mainly because of time constraints and the fact that the visual difference between states can be subtle at a distance, I judged that the added annotation time was not worth it given the scope.

### Issues

Like said before I had nothing to base myself of so the first annotating session was a complete mess and started directly annotating a huge horde of zombies layered on top of eachother and this resulted in the model seeing zombies almost everywhere. This was not viable because having a schizophrenic assistant can only hurt more than do good.

After adjusting the annotations and starting with easier to recognise models, I trained another model and saw that for some reason every garden gnome counted as a Player and almost every Player counted as Zombie, which i could have predicted because there were not that many player instances and a player model is very similar to a zombie model ofcourse.

So I added pictures in the training dataset of only Players running, walking in every direction so all the angles get recognised. This resulted in 300 player instances and a model working much better.

### Final dataset

For the moment i have 371 annotated pictures of the game, with in total a little over 5000 annotations
![<img src="./image.png"/>](image.png)

## Model

I used for this project YOLOv8n model detector, I had never worked with YOLO before nor with computer vision. So after some google work my result was that YOLO was fast and lightweight, so perfect for training on my pc. The training happened locally on my NVIDIA graphics card, the training ran for 100 epochs with patience 20 because it stopped frequently well before reaching the 100th epoch, most of the time it reached the max around the 60th.

This model detects the 7 classes on the screen and passes this data to a script that calculates a risk score based on the size for trees, the amount of zombies, the distance of zombies to the player but also for positive things like escape routes away from danger such as windows and doors. For the flee direction I chose for using the compass instead of the screen because the screen is "tilted" and each corner of the screen represents NE, SE, SW and NW.

To make something of this data + the risk score, a genAI API key was added to formulate advice on a given situation. Unfortunately due to the limitations of the API key, I made genAI write a little script that analyzes the given data and gives a static response.

## Results

My final model had the following results:

[<img src="../runs/detect/models/zomboid_detector/results.png" width="500"/>](../runs/detect/models/zomboid_detector/results.png)

Overall

<table>
<tr> 
<th>Class</th> 
<th>Images</th> 
<th>Instances</th> 
<th>P</th> 
<th>R</th> 
<th>mAP50</th> 
<th>mAP50-95</th> 
</tr>
<tr>
<th>all</th>
<td>18</td>
<td>363</td>
<td>0.73</td>
<td>0.67</td>
<td>0.696</td>
<td>0.425</td>
</tr>
<tr>
<th>Door</th>
<td>10</td>
<td>26</td>
<td>0.7</td>
<td>0.577</td>
<td>0.564</td>
<td>0.336</td>
</tr>
<tr>
<th>Fire</th>
<td>3</td>
<td>37</td>
<td>0.858</td>
<td>0.622</td>
<td>0.727</td>
<td>0.378</td>
</tr>
<tr>
<th>Player</th>
<td>13</td>
<td>15</td>
<td>0.668</td>
<td>0.67</td>
<td>0.583</td>
<td>0.312</td>
</tr>
<tr>
<th>Tree</th>
<td>8</td>
<td>18</td>
<td>0.876</td>
<td>0.787</td>
<td>0.861</td>
<td>0.623</td>
</tr>
<tr>
<th>Zombie</th>
<td>12</td>
<td>55</td>
<td>0.806</td>
<td>0.709</td>
<td>0.785</td>
<td>0.534</td>
</tr>
<tr>
<th>Zombie Dead</th>
<td>9</td>
<td>132</td>
<td>0.592</td>
<td>0.727</td>
<td>0.712</td>
<td>0.4</td>
</tr>
<tr>
<th>Window</th>
<td>11</td>
<td>80</td>
<td>0.608</td>
<td>0.6</td>
<td>0.642</td>
<td>0.395</td>
</tr>
</table>

1. The best performing class is Tree, it was also the most easy to distinct in the pictures and a lot of trees are similar or even identical.

2. The model also produces solid results regarding Zombie class and I am super happy about this result because even with all the data I was scared that zombies would be one of the most difficult to identify.

3. Fire class looks very good, but there are not many cases to evaluate against and it is almost always identical in everry screenshot so it should be easy to identify.

4. Dead zombies have the lowest Precision, so the model flags things as dead zombies that should have been just background. I suspect this is the 2D layering causing bodies and background texture to blend together. On the plus side, Recall is high, so when a dead zombie is actually there, it usually gets found. But at the same time if you flag everything as a dead body you have a high chance of getting the ones that are correct.

5. Window and Door do okay and that is expected because there are a lot of different states of both, variations of both and most of them you can see through so sometimes the model interprets the objects behind the window as part of a window in a different scenario. Doors and windows often have similar structures too so that can cause confusion and make a door count as a window. When that happens often the doorframe gets counted as a window and the actual door counts as a door.

[<img src="../runs/detect/reports/predictions/img_102_jpeg.rf.OUUKLIBdP1oyOvkpfOuv.jpg" width="500"/>](../runs/detect/reports/predictions/img_102_jpeg.rf.OUUKLIBdP1oyOvkpfOuv.jpg)

6. Player is the worst performing class. When looking at the predictions on the images the model sometimes either draws multiple bounding boxes around the same player or draws the player as player aswell as zombie

## Contributions

### Forums and Google

Thanks to the many others who came before me and asked similar questions as me on online forums, I found out about Roboflow, YOLO and questions about batch size, image size, how to improve training or annotating.

### GenAI

I kept generative AI close to me during this project, which I preffered to avoid a little more but I used to to help me with the time constraints and to automate tasks that could steal time and that were not of great importance to the project. Online I read that it is best to have similar names of images like image1,2,3,... so i asked genai to write me [convert](../src/organise_screenshots.py) to rewrite all the names of images to a similar template. I quickly found out that this step can just be skipped.

Another script Claude wrote for me is [convert jfif to jpg](../src/convert_jfif_to_jpg.py), apparantly images on Steam are all in the jfif format and this format is not accepted in Roboflow so I had to reformat them.

I made 2 gameplay videos of me in the game going to dense zombies, inside buildings, forests, etc however converting these videos to pictures would be incredibly time inefficient so Claude wrote me a script to extract pictures from videos: [extract](../src/extract_frames.py)

Because of the time constraints I asked Claude to generate a script that uses my static advice giver that occasionally analyzes a screenshot. I have not been able to test it a lot unfortunately but that is definitely on the agenda when I continue this project

### Me

I wrote the script to train the model and played around with different variables, the script to evaluate the model over test set and print the values, a small script to use a free tier of API of gemini to give advice and a script to automate the static advice script given an image.

## Challenges and Future

The biggest challenge for the model is it has no situational awareness, for example 10 zombies in a small bedroom is more dangerous than 20 zombies in an open field. I did not manage to find a way to make the model determine this. I think to proceed to fix this is either ask the user to provide inside or outside as extra parameter or better and more precise is adding the class Wall but that would be a lot of annotating and a lot of classes extra per picture.
