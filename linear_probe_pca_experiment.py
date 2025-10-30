#!/usr/bin/env python3
"""
Linear Probe PCA Experiment

Trains linear probes on PCA-reduced activations from each layer of GPT-2 Small
to predict linguistic features. Measures mutual information and classification
accuracy across layers.

Tasks:
1. Plurality (Binary): Predict if noun is singular or plural
2. Part of Speech (4-class): Predict POS tag (noun/verb/adjective/adverb)

For each layer (1-11, skipping layer 0 input embeddings):
- Extract 768-dim activations at target token positions
- Apply PCA to reduce to 10 principal components
- Train 3 LogisticRegression probes (for confidence intervals)
- Measure mutual information, accuracy, and F1 score
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.model import ModelLoader
from pos_dataset_generator import generate_pos_dataset


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    logger = logging.getLogger("linear_probe_pca")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def create_plurality_dataset() -> List[Dict]:
    """
    Create dataset for plurality prediction task with 500 unique examples each.

    Returns:
        List of 1000 examples with 'text', 'target_word', and 'label' (0=singular, 1=plural)
    """
    # Generate 500 unique singular examples
    singular_examples = [
        # Animals (50 examples)
        ("The cat sits on the windowsill.", "cat"),
        ("A dog barks at strangers.", "dog"),
        ("The bird sings in the morning.", "bird"),
        ("A horse gallops across the field.", "horse"),
        ("The rabbit hops through the garden.", "rabbit"),
        ("A lion roars in the jungle.", "lion"),
        ("The elephant walks slowly.", "elephant"),
        ("A tiger hunts at night.", "tiger"),
        ("The monkey swings from trees.", "monkey"),
        ("A dolphin swims gracefully.", "dolphin"),
        ("The penguin waddles on ice.", "penguin"),
        ("A bear hibernates in winter.", "bear"),
        ("The snake slithers quietly.", "snake"),
        ("A frog jumps into the pond.", "frog"),
        ("The butterfly emerges from its cocoon.", "butterfly"),
        ("A bee buzzes around flowers.", "bee"),
        ("The ant carries heavy loads.", "ant"),
        ("A spider spins its web.", "spider"),
        ("The fish swims upstream.", "fish"),
        ("A shark patrols the waters.", "shark"),
        ("The whale breaches the surface.", "whale"),
        ("An eagle soars above mountains.", "eagle"),
        ("The owl hoots at midnight.", "owl"),
        ("A parrot repeats words.", "parrot"),
        ("The crow caws loudly.", "crow"),
        ("A seagull flies over the ocean.", "seagull"),
        ("The duck quacks in the pond.", "duck"),
        ("A goose honks aggressively.", "goose"),
        ("The turkey gobbles nervously.", "turkey"),
        ("A chicken pecks at grain.", "chicken"),
        ("The cow moos in the barn.", "cow"),
        ("A pig wallows in mud.", "pig"),
        ("The sheep grazes on grass.", "sheep"),
        ("A goat climbs the rocks.", "goat"),
        ("The donkey brays stubbornly.", "donkey"),
        ("A camel travels through deserts.", "camel"),
        ("The giraffe reaches high branches.", "giraffe"),
        ("A zebra has distinctive stripes.", "zebra"),
        ("The rhino charges forward.", "rhino"),
        ("A hippo rests in water.", "hippo"),
        ("The kangaroo jumps far.", "kangaroo"),
        ("A koala sleeps in trees.", "koala"),
        ("The panda eats bamboo.", "panda"),
        ("A wolf howls at the moon.", "wolf"),
        ("The fox hunts cleverly.", "fox"),
        ("A deer runs through the forest.", "deer"),
        ("The moose has large antlers.", "moose"),
        ("A raccoon searches for food.", "raccoon"),
        ("The squirrel collects nuts.", "squirrel"),
        ("A mouse scurries away.", "mouse"),

        # People and Professions (100 examples)
        ("The student studies diligently.", "student"),
        ("A teacher explains concepts clearly.", "teacher"),
        ("The professor lectures enthusiastically.", "professor"),
        ("A doctor examines patients carefully.", "doctor"),
        ("The nurse administers medication.", "nurse"),
        ("A surgeon performs operations.", "surgeon"),
        ("The dentist cleans teeth.", "dentist"),
        ("A therapist listens attentively.", "therapist"),
        ("The scientist conducts experiments.", "scientist"),
        ("A researcher analyzes data.", "researcher"),
        ("The engineer designs systems.", "engineer"),
        ("A programmer writes code.", "programmer"),
        ("The developer builds applications.", "developer"),
        ("A designer creates graphics.", "designer"),
        ("The artist paints masterpieces.", "artist"),
        ("A musician plays instruments.", "musician"),
        ("The singer performs songs.", "singer"),
        ("A dancer moves gracefully.", "dancer"),
        ("The actor portrays characters.", "actor"),
        ("A director manages productions.", "director"),
        ("The writer composes stories.", "writer"),
        ("An author publishes books.", "author"),
        ("The poet crafts verses.", "poet"),
        ("A journalist reports news.", "journalist"),
        ("The editor reviews manuscripts.", "editor"),
        ("A photographer captures images.", "photographer"),
        ("The chef prepares meals.", "chef"),
        ("A cook follows recipes.", "cook"),
        ("The waiter serves customers.", "waiter"),
        ("A bartender mixes drinks.", "bartender"),
        ("The lawyer argues cases.", "lawyer"),
        ("An attorney represents clients.", "attorney"),
        ("The judge presides over trials.", "judge"),
        ("A politician campaigns actively.", "politician"),
        ("The mayor governs the city.", "mayor"),
        ("A senator proposes legislation.", "senator"),
        ("The officer patrols the streets.", "officer"),
        ("A detective solves crimes.", "detective"),
        ("The firefighter rescues people.", "firefighter"),
        ("A paramedic provides emergency care.", "paramedic"),
        ("The soldier follows orders.", "soldier"),
        ("A pilot flies aircraft.", "pilot"),
        ("The captain commands the ship.", "captain"),
        ("An astronaut explores space.", "astronaut"),
        ("The mechanic repairs vehicles.", "mechanic"),
        ("A plumber fixes pipes.", "plumber"),
        ("The electrician installs wiring.", "electrician"),
        ("A carpenter builds furniture.", "carpenter"),
        ("The architect plans buildings.", "architect"),
        ("A contractor manages construction.", "contractor"),
        ("The farmer grows crops.", "farmer"),
        ("A gardener tends plants.", "gardener"),
        ("The fisherman catches fish.", "fisherman"),
        ("A miner extracts minerals.", "miner"),
        ("The librarian organizes books.", "librarian"),
        ("A clerk files documents.", "clerk"),
        ("The accountant manages finances.", "accountant"),
        ("A banker handles transactions.", "banker"),
        ("The economist studies markets.", "economist"),
        ("A merchant sells goods.", "merchant"),
        ("The cashier processes payments.", "cashier"),
        ("A salesman pitches products.", "salesman"),
        ("The manager supervises teams.", "manager"),
        ("An executive makes decisions.", "executive"),
        ("The entrepreneur starts businesses.", "entrepreneur"),
        ("A consultant provides advice.", "consultant"),
        ("The coach trains athletes.", "coach"),
        ("An instructor teaches classes.", "instructor"),
        ("The trainer guides workouts.", "trainer"),
        ("A guide leads tours.", "guide"),
        ("The translator converts languages.", "translator"),
        ("An interpreter facilitates communication.", "interpreter"),
        ("The secretary schedules appointments.", "secretary"),
        ("An assistant helps with tasks.", "assistant"),
        ("The receptionist greets visitors.", "receptionist"),
        ("A custodian maintains facilities.", "custodian"),
        ("The janitor cleans buildings.", "janitor"),
        ("A guard watches premises.", "guard"),
        ("The volunteer contributes time.", "volunteer"),
        ("An intern learns skills.", "intern"),
        ("The apprentice studies a trade.", "apprentice"),
        ("A craftsman creates items.", "craftsman"),
        ("The tailor sews garments.", "tailor"),
        ("A barber cuts hair.", "barber"),
        ("The hairdresser styles hair.", "hairdresser"),
        ("A beautician applies makeup.", "beautician"),
        ("The optician fits glasses.", "optician"),
        ("A veterinarian treats animals.", "veterinarian"),
        ("The pharmacist dispenses medicine.", "pharmacist"),
        ("A chemist analyzes substances.", "chemist"),
        ("The biologist studies life.", "biologist"),
        ("A physicist explores matter.", "physicist"),
        ("The mathematician solves equations.", "mathematician"),
        ("An astronomer observes stars.", "astronomer"),
        ("The geologist examines rocks.", "geologist"),
        ("A meteorologist predicts weather.", "meteorologist"),
        ("The archaeologist excavates sites.", "archaeologist"),
        ("An anthropologist studies cultures.", "anthropologist"),
        ("The historian documents events.", "historian"),
        ("A philosopher ponders existence.", "philosopher"),

        # Objects and Things (200 examples)
        ("The book contains valuable information.", "book"),
        ("A chair supports people comfortably.", "chair"),
        ("The table holds various items.", "table"),
        ("A desk provides workspace.", "desk"),
        ("The lamp illuminates the room.", "lamp"),
        ("A candle flickers gently.", "candle"),
        ("The door opens inward.", "door"),
        ("A window provides ventilation.", "window"),
        ("The wall stands firmly.", "wall"),
        ("A floor needs cleaning.", "floor"),
        ("The ceiling has been painted.", "ceiling"),
        ("A roof protects from rain.", "roof"),
        ("The house looks welcoming.", "house"),
        ("A building towers impressively.", "building"),
        ("The bridge spans the river.", "bridge"),
        ("A road leads somewhere.", "road"),
        ("The path winds through trees.", "path"),
        ("A street bustles with activity.", "street"),
        ("The car drives smoothly.", "car"),
        ("A truck carries cargo.", "truck"),
        ("The bus transports passengers.", "bus"),
        ("A train arrives punctually.", "train"),
        ("The bicycle leans against the wall.", "bicycle"),
        ("A motorcycle roars loudly.", "motorcycle"),
        ("The airplane flies overhead.", "airplane"),
        ("A helicopter hovers nearby.", "helicopter"),
        ("The boat floats peacefully.", "boat"),
        ("A ship sails the ocean.", "ship"),
        ("The computer processes information.", "computer"),
        ("A phone rings insistently.", "phone"),
        ("The tablet displays content.", "tablet"),
        ("A laptop runs efficiently.", "laptop"),
        ("The keyboard clicks rhythmically.", "keyboard"),
        ("A mouse moves precisely.", "mouse"),
        ("The screen shows results.", "screen"),
        ("A monitor displays graphics.", "monitor"),
        ("The printer produces documents.", "printer"),
        ("A scanner digitizes images.", "scanner"),
        ("The camera captures moments.", "camera"),
        ("A microphone records audio.", "microphone"),
        ("The speaker plays music.", "speaker"),
        ("A headphone delivers sound.", "headphone"),
        ("The television broadcasts programs.", "television"),
        ("A radio receives signals.", "radio"),
        ("The refrigerator keeps food fresh.", "refrigerator"),
        ("An oven bakes food.", "oven"),
        ("The stove heats pots.", "stove"),
        ("A microwave warms meals.", "microwave"),
        ("The toaster browns bread.", "toaster"),
        ("A blender mixes ingredients.", "blender"),
        ("The dishwasher cleans plates.", "dishwasher"),
        ("A washer cleans clothes.", "washer"),
        ("The dryer removes moisture.", "dryer"),
        ("A vacuum removes dirt.", "vacuum"),
        ("The fan circulates air.", "fan"),
        ("An air-conditioner cools spaces.", "air-conditioner"),
        ("The heater warms rooms.", "heater"),
        ("A thermostat regulates temperature.", "thermostat"),
        ("The clock tells time.", "clock"),
        ("A watch shows hours.", "watch"),
        ("The calendar marks dates.", "calendar"),
        ("An alarm rings loudly.", "alarm"),
        ("The bell chimes melodiously.", "bell"),
        ("A whistle sounds sharply.", "whistle"),
        ("The siren wails urgently.", "siren"),
        ("A horn honks repeatedly.", "horn"),
        ("The pen writes smoothly.", "pen"),
        ("A pencil marks paper.", "pencil"),
        ("The marker draws boldly.", "marker"),
        ("A crayon colors brightly.", "crayon"),
        ("The brush paints surfaces.", "brush"),
        ("An eraser removes marks.", "eraser"),
        ("The ruler measures length.", "ruler"),
        ("A compass draws circles.", "compass"),
        ("The calculator computes numbers.", "calculator"),
        ("A notebook stores notes.", "notebook"),
        ("The journal records thoughts.", "journal"),
        ("A diary contains secrets.", "diary"),
        ("The magazine features articles.", "magazine"),
        ("A newspaper reports events.", "newspaper"),
        ("The novel tells stories.", "novel"),
        ("A textbook explains concepts.", "textbook"),
        ("The dictionary defines words.", "dictionary"),
        ("An encyclopedia provides knowledge.", "encyclopedia"),
        ("The atlas shows maps.", "atlas"),
        ("A manual gives instructions.", "manual"),
        ("The recipe describes cooking.", "recipe"),
        ("A map indicates locations.", "map"),
        ("The chart displays data.", "chart"),
        ("A graph visualizes trends.", "graph"),
        ("The diagram illustrates processes.", "diagram"),
        ("A picture shows scenes.", "picture"),
        ("The painting depicts beauty.", "painting"),
        ("A photograph freezes moments.", "photograph"),
        ("The sculpture stands prominently.", "sculpture"),
        ("A statue represents figures.", "statue"),
        ("The monument commemorates events.", "monument"),
        ("A trophy symbolizes achievement.", "trophy"),
        ("The medal honors excellence.", "medal"),
        ("A prize rewards winners.", "prize"),
        ("The certificate confirms completion.", "certificate"),
        ("A diploma proves graduation.", "diploma"),
        ("The license permits activities.", "license"),
        ("A passport enables travel.", "passport"),
        ("The ticket grants entry.", "ticket"),
        ("A receipt confirms payment.", "receipt"),
        ("The invoice requests payment.", "invoice"),
        ("A contract binds parties.", "contract"),
        ("The agreement establishes terms.", "agreement"),
        ("A document contains information.", "document"),
        ("The file organizes data.", "file"),
        ("A folder holds papers.", "folder"),
        ("The envelope contains letters.", "envelope"),
        ("A package arrives today.", "package"),
        ("The box stores items.", "box"),
        ("A container holds contents.", "container"),
        ("The basket carries goods.", "basket"),
        ("A bag holds belongings.", "bag"),
        ("The suitcase contains clothes.", "suitcase"),
        ("A backpack carries supplies.", "backpack"),
        ("The purse holds essentials.", "purse"),
        ("A wallet stores money.", "wallet"),
        ("The coin has value.", "coin"),
        ("A bill represents currency.", "bill"),
        ("The card enables transactions.", "card"),
        ("A key unlocks doors.", "key"),
        ("The lock secures entrances.", "lock"),
        ("A chain connects links.", "chain"),
        ("The rope ties objects.", "rope"),
        ("A string binds packages.", "string"),
        ("The wire conducts electricity.", "wire"),
        ("A cable transmits signals.", "cable"),
        ("The pipe carries water.", "pipe"),
        ("A hose sprays liquid.", "hose"),
        ("The tube contains paste.", "tube"),
        ("A bottle holds beverages.", "bottle"),
        ("The jar preserves food.", "jar"),
        ("A can stores goods.", "can"),
        ("The cup holds drinks.", "cup"),
        ("A glass contains liquid.", "glass"),
        ("The mug keeps coffee warm.", "mug"),
        ("A bowl contains soup.", "bowl"),
        ("The plate holds food.", "plate"),
        ("A dish serves meals.", "dish"),
        ("The spoon stirs ingredients.", "spoon"),
        ("A fork picks food.", "fork"),
        ("The knife cuts precisely.", "knife"),
        ("A chopstick helps eating.", "chopstick"),
        ("The napkin wipes hands.", "napkin"),
        ("A towel dries surfaces.", "towel"),
        ("The blanket provides warmth.", "blanket"),
        ("A pillow supports heads.", "pillow"),
        ("The mattress offers comfort.", "mattress"),
        ("A bed facilitates sleep.", "bed"),
        ("The couch seats people.", "couch"),
        ("A sofa provides seating.", "sofa"),
        ("The bench offers rest.", "bench"),
        ("A stool elevates height.", "stool"),

        # Nature and Places (150 examples)
        ("The mountain rises majestically.", "mountain"),
        ("A hill slopes gently.", "hill"),
        ("The valley stretches wide.", "valley"),
        ("A canyon cuts deep.", "canyon"),
        ("The cliff drops sharply.", "cliff"),
        ("A cave provides shelter.", "cave"),
        ("The river flows steadily.", "river"),
        ("A stream babbles softly.", "stream"),
        ("The lake reflects sky.", "lake"),
        ("A pond teems with life.", "pond"),
        ("The ocean crashes powerfully.", "ocean"),
        ("A sea extends endlessly.", "sea"),
        ("The wave crashes ashore.", "wave"),
        ("A tide rises predictably.", "tide"),
        ("The beach attracts visitors.", "beach"),
        ("A shore meets water.", "shore"),
        ("The island stands isolated.", "island"),
        ("A peninsula juts outward.", "peninsula"),
        ("The forest grows densely.", "forest"),
        ("A jungle thrives tropically.", "jungle"),
        ("The tree provides oxygen.", "tree"),
        ("A bush grows thickly.", "bush"),
        ("The shrub needs trimming.", "shrub"),
        ("A plant photosynthesizes daily.", "plant"),
        ("The flower blooms beautifully.", "flower"),
        ("A seed germinates slowly.", "seed"),
        ("The leaf changes color.", "leaf"),
        ("A branch extends outward.", "branch"),
        ("The trunk supports limbs.", "trunk"),
        ("A root anchors firmly.", "root"),
        ("The grass grows quickly.", "grass"),
        ("A weed spreads rapidly.", "weed"),
        ("The field yields crops.", "field"),
        ("A meadow blooms colorfully.", "meadow"),
        ("The prairie stretches far.", "prairie"),
        ("A plain extends flatly.", "plain"),
        ("The desert lacks water.", "desert"),
        ("A dune shifts constantly.", "dune"),
        ("The oasis offers refuge.", "oasis"),
        ("A tundra remains frozen.", "tundra"),
        ("The glacier moves slowly.", "glacier"),
        ("An iceberg floats dangerously.", "iceberg"),
        ("The snow falls gently.", "snow"),
        ("A snowflake drifts down.", "snowflake"),
        ("The ice forms overnight.", "ice"),
        ("A cloud drifts lazily.", "cloud"),
        ("The sky appears blue.", "sky"),
        ("A star twinkles brightly.", "star"),
        ("The sun shines warmly.", "sun"),
        ("A moon illuminates night.", "moon"),
        ("The planet orbits regularly.", "planet"),
        ("A meteor streaks across.", "meteor"),
        ("The comet appears rarely.", "comet"),
        ("A constellation forms patterns.", "constellation"),
        ("The galaxy contains billions.", "galaxy"),
        ("A universe expands infinitely.", "universe"),
        ("The atmosphere protects Earth.", "atmosphere"),
        ("A wind blows strongly.", "wind"),
        ("The breeze feels refreshing.", "breeze"),
        ("A storm approaches quickly.", "storm"),
        ("The rain falls steadily.", "rain"),
        ("A raindrop splashes down.", "raindrop"),
        ("The thunder rumbles loudly.", "thunder"),
        ("A lightning strikes suddenly.", "lightning"),
        ("The fog obscures vision.", "fog"),
        ("A mist rises gently.", "mist"),
        ("The dew forms overnight.", "dew"),
        ("A frost covers surfaces.", "frost"),
        ("The rainbow arcs beautifully.", "rainbow"),
        ("A tornado spins violently.", "tornado"),
        ("The hurricane devastates areas.", "hurricane"),
        ("A cyclone rotates powerfully.", "cyclone"),
        ("The earthquake shakes ground.", "earthquake"),
        ("A volcano erupts violently.", "volcano"),
        ("The lava flows hot.", "lava"),
        ("A rock sits motionless.", "rock"),
        ("The stone remains solid.", "stone"),
        ("A pebble skips smoothly.", "pebble"),
        ("The boulder blocks paths.", "boulder"),
        ("A mineral contains elements.", "mineral"),
        ("The crystal reflects light.", "crystal"),
        ("A gem sparkles brilliantly.", "gem"),
        ("The diamond shines forever.", "diamond"),
        ("A pearl forms naturally.", "pearl"),
        ("The gold gleams richly.", "gold"),
        ("A silver tarnishes slowly.", "silver"),
        ("The copper conducts well.", "copper"),
        ("An iron rusts easily.", "iron"),
        ("The steel remains strong.", "steel"),
        ("A metal conducts heat.", "metal"),
        ("The wood burns slowly.", "wood"),
        ("A log fuels fires.", "log"),
        ("The coal provides energy.", "coal"),
        ("An oil lubricates machinery.", "oil"),
        ("The gas expands freely.", "gas"),
        ("A liquid flows easily.", "liquid"),
        ("The water hydrates bodies.", "water"),
        ("A drop falls softly.", "drop"),
        ("The fire burns brightly.", "fire"),
        ("A flame flickers constantly.", "flame"),
        ("The smoke rises upward.", "smoke"),
        ("An ash settles down.", "ash"),
        ("The ember glows dimly.", "ember"),
        ("A spark ignites fuel.", "spark"),
        ("The explosion occurs suddenly.", "explosion"),
        ("A sound travels through air.", "sound"),
        ("The noise disturbs peace.", "noise"),
        ("A silence feels peaceful.", "silence"),
        ("The echo repeats back.", "echo"),
        ("A voice speaks clearly.", "voice"),
        ("The tone conveys emotion.", "tone"),
        ("A pitch varies widely.", "pitch"),
        ("The rhythm beats steadily.", "rhythm"),
        ("A melody sounds pleasant.", "melody"),
        ("The harmony blends perfectly.", "harmony"),
        ("A chord resonates deeply.", "chord"),
        ("The note holds long.", "note"),
        ("A beat pulses regularly.", "beat"),
        ("The tempo changes frequently.", "tempo"),
        ("A song plays repeatedly.", "song"),
        ("The music fills space.", "music"),
        ("A tune sticks mentally.", "tune"),
        ("The symphony performs magnificently.", "symphony"),
        ("An orchestra plays together.", "orchestra"),
        ("The band performs live.", "band"),
        ("A choir sings harmoniously.", "choir"),
        ("The audience applauds loudly.", "audience"),
        ("A crowd gathers quickly.", "crowd"),
        ("The group meets regularly.", "group"),
        ("A team works together.", "team"),
        ("The club welcomes members.", "club"),
        ("An organization serves communities.", "organization"),
        ("The company employs workers.", "company"),
        ("A business operates daily.", "business"),
        ("The store sells products.", "store"),
        ("A shop offers services.", "shop"),
        ("The market bustles actively.", "market"),
        ("A mall attracts shoppers.", "mall"),
        ("The restaurant serves meals.", "restaurant"),
        ("A cafe offers drinks.", "cafe"),
        ("The bar serves beverages.", "bar"),
        ("A hotel accommodates guests.", "hotel"),
        ("The hospital treats patients.", "hospital"),
        ("A clinic provides care.", "clinic"),
        ("The school educates students.", "school"),
        ("A university offers degrees.", "university"),
        ("The college prepares graduates.", "college"),
        ("A library lends books.", "library"),
        ("The museum displays artifacts.", "museum"),
        ("A gallery exhibits art.", "gallery"),
        ("The theater shows performances.", "theater"),
        ("A cinema screens films.", "cinema"),
    ]

    # Generate 500 unique plural examples (matching singular examples)
    plural_examples = [
        # Animals (50 examples)
        ("The cats sit on the windowsill.", "cats"),
        ("Dogs bark at strangers.", "Dogs"),
        ("The birds sing in the morning.", "birds"),
        ("Horses gallop across the field.", "Horses"),
        ("The rabbits hop through the garden.", "rabbits"),
        ("Lions roar in the jungle.", "Lions"),
        ("The elephants walk slowly.", "elephants"),
        ("Tigers hunt at night.", "Tigers"),
        ("The monkeys swing from trees.", "monkeys"),
        ("Dolphins swim gracefully.", "Dolphins"),
        ("The penguins waddle on ice.", "penguins"),
        ("Bears hibernate in winter.", "Bears"),
        ("The snakes slither quietly.", "snakes"),
        ("Frogs jump into the pond.", "Frogs"),
        ("The butterflies emerge from their cocoons.", "butterflies"),
        ("Bees buzz around flowers.", "Bees"),
        ("The ants carry heavy loads.", "ants"),
        ("Spiders spin their webs.", "Spiders"),
        ("The fish swim upstream.", "fish"),
        ("Sharks patrol the waters.", "Sharks"),
        ("The whales breach the surface.", "whales"),
        ("Eagles soar above mountains.", "Eagles"),
        ("The owls hoot at midnight.", "owls"),
        ("Parrots repeat words.", "Parrots"),
        ("The crows caw loudly.", "crows"),
        ("Seagulls fly over the ocean.", "Seagulls"),
        ("The ducks quack in the pond.", "ducks"),
        ("Geese honk aggressively.", "Geese"),
        ("The turkeys gobble nervously.", "turkeys"),
        ("Chickens peck at grain.", "Chickens"),
        ("The cows moo in the barn.", "cows"),
        ("Pigs wallow in mud.", "Pigs"),
        ("The sheep graze on grass.", "sheep"),
        ("Goats climb the rocks.", "Goats"),
        ("The donkeys bray stubbornly.", "donkeys"),
        ("Camels travel through deserts.", "Camels"),
        ("The giraffes reach high branches.", "giraffes"),
        ("Zebras have distinctive stripes.", "Zebras"),
        ("The rhinos charge forward.", "rhinos"),
        ("Hippos rest in water.", "Hippos"),
        ("The kangaroos jump far.", "kangaroos"),
        ("Koalas sleep in trees.", "Koalas"),
        ("The pandas eat bamboo.", "pandas"),
        ("Wolves howl at the moon.", "Wolves"),
        ("The foxes hunt cleverly.", "foxes"),
        ("Deer run through the forest.", "Deer"),
        ("The moose have large antlers.", "moose"),
        ("Raccoons search for food.", "Raccoons"),
        ("The squirrels collect nuts.", "squirrels"),
        ("Mice scurry away.", "Mice"),

        # People and Professions (100 examples)
        ("The students study diligently.", "students"),
        ("Teachers explain concepts clearly.", "Teachers"),
        ("The professors lecture enthusiastically.", "professors"),
        ("Doctors examine patients carefully.", "Doctors"),
        ("The nurses administer medication.", "nurses"),
        ("Surgeons perform operations.", "Surgeons"),
        ("The dentists clean teeth.", "dentists"),
        ("Therapists listen attentively.", "Therapists"),
        ("The scientists conduct experiments.", "scientists"),
        ("Researchers analyze data.", "Researchers"),
        ("The engineers design systems.", "engineers"),
        ("Programmers write code.", "Programmers"),
        ("The developers build applications.", "developers"),
        ("Designers create graphics.", "Designers"),
        ("The artists paint masterpieces.", "artists"),
        ("Musicians play instruments.", "Musicians"),
        ("The singers perform songs.", "singers"),
        ("Dancers move gracefully.", "Dancers"),
        ("The actors portray characters.", "actors"),
        ("Directors manage productions.", "Directors"),
        ("The writers compose stories.", "writers"),
        ("Authors publish books.", "Authors"),
        ("The poets craft verses.", "poets"),
        ("Journalists report news.", "Journalists"),
        ("The editors review manuscripts.", "editors"),
        ("Photographers capture images.", "Photographers"),
        ("The chefs prepare meals.", "chefs"),
        ("Cooks follow recipes.", "Cooks"),
        ("The waiters serve customers.", "waiters"),
        ("Bartenders mix drinks.", "Bartenders"),
        ("The lawyers argue cases.", "lawyers"),
        ("Attorneys represent clients.", "Attorneys"),
        ("The judges preside over trials.", "judges"),
        ("Politicians campaign actively.", "Politicians"),
        ("The mayors govern the cities.", "mayors"),
        ("Senators propose legislation.", "Senators"),
        ("The officers patrol the streets.", "officers"),
        ("Detectives solve crimes.", "Detectives"),
        ("The firefighters rescue people.", "firefighters"),
        ("Paramedics provide emergency care.", "Paramedics"),
        ("The soldiers follow orders.", "soldiers"),
        ("Pilots fly aircraft.", "Pilots"),
        ("The captains command the ships.", "captains"),
        ("Astronauts explore space.", "Astronauts"),
        ("The mechanics repair vehicles.", "mechanics"),
        ("Plumbers fix pipes.", "Plumbers"),
        ("The electricians install wiring.", "electricians"),
        ("Carpenters build furniture.", "Carpenters"),
        ("The architects plan buildings.", "architects"),
        ("Contractors manage construction.", "Contractors"),
        ("The farmers grow crops.", "farmers"),
        ("Gardeners tend plants.", "Gardeners"),
        ("The fishermen catch fish.", "fishermen"),
        ("Miners extract minerals.", "Miners"),
        ("The librarians organize books.", "librarians"),
        ("Clerks file documents.", "Clerks"),
        ("The accountants manage finances.", "accountants"),
        ("Bankers handle transactions.", "Bankers"),
        ("The economists study markets.", "economists"),
        ("Merchants sell goods.", "Merchants"),
        ("The cashiers process payments.", "cashiers"),
        ("Salesmen pitch products.", "Salesmen"),
        ("The managers supervise teams.", "managers"),
        ("Executives make decisions.", "Executives"),
        ("The entrepreneurs start businesses.", "entrepreneurs"),
        ("Consultants provide advice.", "Consultants"),
        ("The coaches train athletes.", "coaches"),
        ("Instructors teach classes.", "Instructors"),
        ("The trainers guide workouts.", "trainers"),
        ("Guides lead tours.", "Guides"),
        ("The translators convert languages.", "translators"),
        ("Interpreters facilitate communication.", "Interpreters"),
        ("The secretaries schedule appointments.", "secretaries"),
        ("Assistants help with tasks.", "Assistants"),
        ("The receptionists greet visitors.", "receptionists"),
        ("Custodians maintain facilities.", "Custodians"),
        ("The janitors clean buildings.", "janitors"),
        ("Guards watch premises.", "Guards"),
        ("The volunteers contribute time.", "volunteers"),
        ("Interns learn skills.", "Interns"),
        ("The apprentices study trades.", "apprentices"),
        ("Craftsmen create items.", "Craftsmen"),
        ("The tailors sew garments.", "tailors"),
        ("Barbers cut hair.", "Barbers"),
        ("The hairdressers style hair.", "hairdressers"),
        ("Beauticians apply makeup.", "Beauticians"),
        ("The opticians fit glasses.", "opticians"),
        ("Veterinarians treat animals.", "Veterinarians"),
        ("The pharmacists dispense medicine.", "pharmacists"),
        ("Chemists analyze substances.", "Chemists"),
        ("The biologists study life.", "biologists"),
        ("Physicists explore matter.", "Physicists"),
        ("The mathematicians solve equations.", "mathematicians"),
        ("Astronomers observe stars.", "Astronomers"),
        ("The geologists examine rocks.", "geologists"),
        ("Meteorologists predict weather.", "Meteorologists"),
        ("The archaeologists excavate sites.", "archaeologists"),
        ("Anthropologists study cultures.", "Anthropologists"),
        ("The historians document events.", "historians"),
        ("Philosophers ponder existence.", "Philosophers"),

        # Objects and Things (200 examples)
        ("The books contain valuable information.", "books"),
        ("Chairs support people comfortably.", "Chairs"),
        ("The tables hold various items.", "tables"),
        ("Desks provide workspace.", "Desks"),
        ("The lamps illuminate the room.", "lamps"),
        ("Candles flicker gently.", "Candles"),
        ("The doors open inward.", "doors"),
        ("Windows provide ventilation.", "Windows"),
        ("The walls stand firmly.", "walls"),
        ("Floors need cleaning.", "Floors"),
        ("The ceilings have been painted.", "ceilings"),
        ("Roofs protect from rain.", "Roofs"),
        ("The houses look welcoming.", "houses"),
        ("Buildings tower impressively.", "Buildings"),
        ("The bridges span the rivers.", "bridges"),
        ("Roads lead somewhere.", "Roads"),
        ("The paths wind through trees.", "paths"),
        ("Streets bustle with activity.", "Streets"),
        ("The cars drive smoothly.", "cars"),
        ("Trucks carry cargo.", "Trucks"),
        ("The buses transport passengers.", "buses"),
        ("Trains arrive punctually.", "Trains"),
        ("The bicycles lean against the wall.", "bicycles"),
        ("Motorcycles roar loudly.", "Motorcycles"),
        ("The airplanes fly overhead.", "airplanes"),
        ("Helicopters hover nearby.", "Helicopters"),
        ("The boats float peacefully.", "boats"),
        ("Ships sail the ocean.", "Ships"),
        ("The computers process information.", "computers"),
        ("Phones ring insistently.", "Phones"),
        ("The tablets display content.", "tablets"),
        ("Laptops run efficiently.", "Laptops"),
        ("The keyboards click rhythmically.", "keyboards"),
        ("Mice move precisely.", "Mice"),
        ("The screens show results.", "screens"),
        ("Monitors display graphics.", "Monitors"),
        ("The printers produce documents.", "printers"),
        ("Scanners digitize images.", "Scanners"),
        ("The cameras capture moments.", "cameras"),
        ("Microphones record audio.", "Microphones"),
        ("The speakers play music.", "speakers"),
        ("Headphones deliver sound.", "Headphones"),
        ("The televisions broadcast programs.", "televisions"),
        ("Radios receive signals.", "Radios"),
        ("The refrigerators keep food fresh.", "refrigerators"),
        ("Ovens bake food.", "Ovens"),
        ("The stoves heat pots.", "stoves"),
        ("Microwaves warm meals.", "Microwaves"),
        ("The toasters brown bread.", "toasters"),
        ("Blenders mix ingredients.", "Blenders"),
        ("The dishwashers clean plates.", "dishwashers"),
        ("Washers clean clothes.", "Washers"),
        ("The dryers remove moisture.", "dryers"),
        ("Vacuums remove dirt.", "Vacuums"),
        ("The fans circulate air.", "fans"),
        ("Air-conditioners cool spaces.", "Air-conditioners"),
        ("The heaters warm rooms.", "heaters"),
        ("Thermostats regulate temperature.", "Thermostats"),
        ("The clocks tell time.", "clocks"),
        ("Watches show hours.", "Watches"),
        ("The calendars mark dates.", "calendars"),
        ("Alarms ring loudly.", "Alarms"),
        ("The bells chime melodiously.", "bells"),
        ("Whistles sound sharply.", "Whistles"),
        ("The sirens wail urgently.", "sirens"),
        ("Horns honk repeatedly.", "Horns"),
        ("The pens write smoothly.", "pens"),
        ("Pencils mark paper.", "Pencils"),
        ("The markers draw boldly.", "markers"),
        ("Crayons color brightly.", "Crayons"),
        ("The brushes paint surfaces.", "brushes"),
        ("Erasers remove marks.", "Erasers"),
        ("The rulers measure length.", "rulers"),
        ("Compasses draw circles.", "Compasses"),
        ("The calculators compute numbers.", "calculators"),
        ("Notebooks store notes.", "Notebooks"),
        ("The journals record thoughts.", "journals"),
        ("Diaries contain secrets.", "Diaries"),
        ("The magazines feature articles.", "magazines"),
        ("Newspapers report events.", "Newspapers"),
        ("The novels tell stories.", "novels"),
        ("Textbooks explain concepts.", "Textbooks"),
        ("The dictionaries define words.", "dictionaries"),
        ("Encyclopedias provide knowledge.", "Encyclopedias"),
        ("The atlases show maps.", "atlases"),
        ("Manuals give instructions.", "Manuals"),
        ("The recipes describe cooking.", "recipes"),
        ("Maps indicate locations.", "Maps"),
        ("The charts display data.", "charts"),
        ("Graphs visualize trends.", "Graphs"),
        ("The diagrams illustrate processes.", "diagrams"),
        ("Pictures show scenes.", "Pictures"),
        ("The paintings depict beauty.", "paintings"),
        ("Photographs freeze moments.", "Photographs"),
        ("The sculptures stand prominently.", "sculptures"),
        ("Statues represent figures.", "Statues"),
        ("The monuments commemorate events.", "monuments"),
        ("Trophies symbolize achievement.", "Trophies"),
        ("The medals honor excellence.", "medals"),
        ("Prizes reward winners.", "Prizes"),
        ("The certificates confirm completion.", "certificates"),
        ("Diplomas prove graduation.", "Diplomas"),
        ("The licenses permit activities.", "licenses"),
        ("Passports enable travel.", "Passports"),
        ("The tickets grant entry.", "tickets"),
        ("Receipts confirm payment.", "Receipts"),
        ("The invoices request payment.", "invoices"),
        ("Contracts bind parties.", "Contracts"),
        ("The agreements establish terms.", "agreements"),
        ("Documents contain information.", "Documents"),
        ("The files organize data.", "files"),
        ("Folders hold papers.", "Folders"),
        ("The envelopes contain letters.", "envelopes"),
        ("Packages arrive today.", "Packages"),
        ("The boxes store items.", "boxes"),
        ("Containers hold contents.", "Containers"),
        ("The baskets carry goods.", "baskets"),
        ("Bags hold belongings.", "Bags"),
        ("The suitcases contain clothes.", "suitcases"),
        ("Backpacks carry supplies.", "Backpacks"),
        ("The purses hold essentials.", "purses"),
        ("Wallets store money.", "Wallets"),
        ("The coins have value.", "coins"),
        ("Bills represent currency.", "Bills"),
        ("The cards enable transactions.", "cards"),
        ("Keys unlock doors.", "Keys"),
        ("The locks secure entrances.", "locks"),
        ("Chains connect links.", "Chains"),
        ("The ropes tie objects.", "ropes"),
        ("Strings bind packages.", "Strings"),
        ("The wires conduct electricity.", "wires"),
        ("Cables transmit signals.", "Cables"),
        ("The pipes carry water.", "pipes"),
        ("Hoses spray liquid.", "Hoses"),
        ("The tubes contain paste.", "tubes"),
        ("Bottles hold beverages.", "Bottles"),
        ("The jars preserve food.", "jars"),
        ("Cans store goods.", "Cans"),
        ("The cups hold drinks.", "cups"),
        ("Glasses contain liquid.", "Glasses"),
        ("The mugs keep coffee warm.", "mugs"),
        ("Bowls contain soup.", "Bowls"),
        ("The plates hold food.", "plates"),
        ("Dishes serve meals.", "Dishes"),
        ("The spoons stir ingredients.", "spoons"),
        ("Forks pick food.", "Forks"),
        ("The knives cut precisely.", "knives"),
        ("Chopsticks help eating.", "Chopsticks"),
        ("The napkins wipe hands.", "napkins"),
        ("Towels dry surfaces.", "Towels"),
        ("The blankets provide warmth.", "blankets"),
        ("Pillows support heads.", "Pillows"),
        ("The mattresses offer comfort.", "mattresses"),
        ("Beds facilitate sleep.", "Beds"),
        ("The couches seat people.", "couches"),
        ("Sofas provide seating.", "Sofas"),
        ("The benches offer rest.", "benches"),
        ("Stools elevate height.", "Stools"),

        # Nature and Places (150 examples)
        ("The mountains rise majestically.", "mountains"),
        ("Hills slope gently.", "Hills"),
        ("The valleys stretch wide.", "valleys"),
        ("Canyons cut deep.", "Canyons"),
        ("The cliffs drop sharply.", "cliffs"),
        ("Caves provide shelter.", "Caves"),
        ("The rivers flow steadily.", "rivers"),
        ("Streams babble softly.", "Streams"),
        ("The lakes reflect sky.", "lakes"),
        ("Ponds teem with life.", "Ponds"),
        ("The oceans crash powerfully.", "oceans"),
        ("Seas extend endlessly.", "Seas"),
        ("The waves crash ashore.", "waves"),
        ("Tides rise predictably.", "Tides"),
        ("The beaches attract visitors.", "beaches"),
        ("Shores meet water.", "Shores"),
        ("The islands stand isolated.", "islands"),
        ("Peninsulas jut outward.", "Peninsulas"),
        ("The forests grow densely.", "forests"),
        ("Jungles thrive tropically.", "Jungles"),
        ("The trees provide oxygen.", "trees"),
        ("Bushes grow thickly.", "Bushes"),
        ("The shrubs need trimming.", "shrubs"),
        ("Plants photosynthesize daily.", "Plants"),
        ("The flowers bloom beautifully.", "flowers"),
        ("Seeds germinate slowly.", "Seeds"),
        ("The leaves change color.", "leaves"),
        ("Branches extend outward.", "Branches"),
        ("The trunks support limbs.", "trunks"),
        ("Roots anchor firmly.", "Roots"),
        ("The grasses grow quickly.", "grasses"),
        ("Weeds spread rapidly.", "Weeds"),
        ("The fields yield crops.", "fields"),
        ("Meadows bloom colorfully.", "Meadows"),
        ("The prairies stretch far.", "prairies"),
        ("Plains extend flatly.", "Plains"),
        ("The deserts lack water.", "deserts"),
        ("Dunes shift constantly.", "Dunes"),
        ("The oases offer refuge.", "oases"),
        ("Tundras remain frozen.", "Tundras"),
        ("The glaciers move slowly.", "glaciers"),
        ("Icebergs float dangerously.", "Icebergs"),
        ("The snows fall gently.", "snows"),
        ("Snowflakes drift down.", "Snowflakes"),
        ("The ices form overnight.", "ices"),
        ("Clouds drift lazily.", "Clouds"),
        ("The skies appear blue.", "skies"),
        ("Stars twinkle brightly.", "Stars"),
        ("The suns shine warmly.", "suns"),
        ("Moons illuminate night.", "Moons"),
        ("The planets orbit regularly.", "planets"),
        ("Meteors streak across.", "Meteors"),
        ("The comets appear rarely.", "comets"),
        ("Constellations form patterns.", "Constellations"),
        ("The galaxies contain billions.", "galaxies"),
        ("Universes expand infinitely.", "Universes"),
        ("The atmospheres protect Earth.", "atmospheres"),
        ("Winds blow strongly.", "Winds"),
        ("The breezes feel refreshing.", "breezes"),
        ("Storms approach quickly.", "Storms"),
        ("The rains fall steadily.", "rains"),
        ("Raindrops splash down.", "Raindrops"),
        ("The thunders rumble loudly.", "thunders"),
        ("Lightnings strike suddenly.", "Lightnings"),
        ("The fogs obscure vision.", "fogs"),
        ("Mists rise gently.", "Mists"),
        ("The dews form overnight.", "dews"),
        ("Frosts cover surfaces.", "Frosts"),
        ("The rainbows arc beautifully.", "rainbows"),
        ("Tornadoes spin violently.", "Tornadoes"),
        ("The hurricanes devastate areas.", "hurricanes"),
        ("Cyclones rotate powerfully.", "Cyclones"),
        ("The earthquakes shake ground.", "earthquakes"),
        ("Volcanoes erupt violently.", "Volcanoes"),
        ("The lavas flow hot.", "lavas"),
        ("Rocks sit motionless.", "Rocks"),
        ("The stones remain solid.", "stones"),
        ("Pebbles skip smoothly.", "Pebbles"),
        ("The boulders block paths.", "boulders"),
        ("Minerals contain elements.", "Minerals"),
        ("The crystals reflect light.", "crystals"),
        ("Gems sparkle brilliantly.", "Gems"),
        ("The diamonds shine forever.", "diamonds"),
        ("Pearls form naturally.", "Pearls"),
        ("The golds gleam richly.", "golds"),
        ("Silvers tarnish slowly.", "Silvers"),
        ("The coppers conduct well.", "coppers"),
        ("Irons rust easily.", "Irons"),
        ("The steels remain strong.", "steels"),
        ("Metals conduct heat.", "Metals"),
        ("The woods burn slowly.", "woods"),
        ("Logs fuel fires.", "Logs"),
        ("The coals provide energy.", "coals"),
        ("Oils lubricate machinery.", "Oils"),
        ("The gases expand freely.", "gases"),
        ("Liquids flow easily.", "Liquids"),
        ("The waters hydrate bodies.", "waters"),
        ("Drops fall softly.", "Drops"),
        ("The fires burn brightly.", "fires"),
        ("Flames flicker constantly.", "Flames"),
        ("The smokes rise upward.", "smokes"),
        ("Ashes settle down.", "Ashes"),
        ("The embers glow dimly.", "embers"),
        ("Sparks ignite fuel.", "Sparks"),
        ("The explosions occur suddenly.", "explosions"),
        ("Sounds travel through air.", "Sounds"),
        ("The noises disturb peace.", "noises"),
        ("Silences feel peaceful.", "Silences"),
        ("The echoes repeat back.", "echoes"),
        ("Voices speak clearly.", "Voices"),
        ("The tones convey emotion.", "tones"),
        ("Pitches vary widely.", "Pitches"),
        ("The rhythms beat steadily.", "rhythms"),
        ("Melodies sound pleasant.", "Melodies"),
        ("The harmonies blend perfectly.", "harmonies"),
        ("Chords resonate deeply.", "Chords"),
        ("The notes hold long.", "notes"),
        ("Beats pulse regularly.", "Beats"),
        ("The tempos change frequently.", "tempos"),
        ("Songs play repeatedly.", "Songs"),
        ("The musics fill space.", "musics"),
        ("Tunes stick mentally.", "Tunes"),
        ("The symphonies perform magnificently.", "symphonies"),
        ("Orchestras play together.", "Orchestras"),
        ("The bands perform live.", "bands"),
        ("Choirs sing harmoniously.", "Choirs"),
        ("The audiences applaud loudly.", "audiences"),
        ("Crowds gather quickly.", "Crowds"),
        ("The groups meet regularly.", "groups"),
        ("Teams work together.", "Teams"),
        ("The clubs welcome members.", "clubs"),
        ("Organizations serve communities.", "Organizations"),
        ("The companies employ workers.", "companies"),
        ("Businesses operate daily.", "Businesses"),
        ("The stores sell products.", "stores"),
        ("Shops offer services.", "Shops"),
        ("The markets bustle actively.", "markets"),
        ("Malls attract shoppers.", "Malls"),
        ("The restaurants serve meals.", "restaurants"),
        ("Cafes offer drinks.", "Cafes"),
        ("The bars serve beverages.", "bars"),
        ("Hotels accommodate guests.", "Hotels"),
        ("The hospitals treat patients.", "hospitals"),
        ("Clinics provide care.", "Clinics"),
        ("The schools educate students.", "schools"),
        ("Universities offer degrees.", "Universities"),
        ("The colleges prepare graduates.", "colleges"),
        ("Libraries lend books.", "Libraries"),
        ("The museums display artifacts.", "museums"),
        ("Galleries exhibit art.", "Galleries"),
        ("The theaters show performances.", "theaters"),
        ("Cinemas screen films.", "Cinemas"),
    ]

    dataset = []

    # Add all unique singular examples
    for text, target in singular_examples:
        dataset.append({'text': text, 'target_word': target, 'label': 0})

    # Add all unique plural examples
    for text, target in plural_examples:
        dataset.append({'text': text, 'target_word': target, 'label': 1})

    return dataset


def create_pos_dataset() -> List[Dict]:
    """
    Create dataset for part-of-speech prediction task.

    Returns 200 unique examples for each POS category (800 total).

    Returns:
        List of 800 examples with 'text', 'target_word', and 'label'
        (0=noun, 1=verb, 2=adjective, 3=adverb)
    """
    return generate_pos_dataset()


def find_target_token_position(
    tokens: torch.Tensor,
    tokenizer,
    text: str,
    target_word: str
) -> int:
    """
    Find the last token position of the target word in the tokenized text.

    Args:
        tokens: Tokenized input (1, seq_len)
        tokenizer: Model tokenizer
        text: Original text
        target_word: Target word to find

    Returns:
        Position index of the last token of the target word
    """
    # Decode all tokens
    token_strs = [tokenizer.decode([t]) for t in tokens[0]]

    # Find target word in original text
    target_start = text.lower().find(target_word.lower())
    if target_start == -1:
        raise ValueError(f"Target word '{target_word}' not found in text '{text}'")

    # Reconstruct text from tokens to find position
    current_pos = 0
    reconstructed = ""

    for i, token_str in enumerate(token_strs):
        # Update reconstructed text
        reconstructed += token_str
        current_pos = len(reconstructed)

        # Check if we've passed the end of the target word
        if current_pos >= target_start + len(target_word):
            return i

    # If we didn't find it, return the last token
    return len(token_strs) - 1


def extract_activations(
    model,
    examples: List[Dict],
    layer: int,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract activations at target token positions for all examples.

    Args:
        model: HookedTransformer model
        examples: List of examples with 'text', 'target_word', 'label'
        layer: Layer index to extract from
        logger: Logger instance

    Returns:
        Tuple of (activations, labels) as numpy arrays
        activations: (n_examples, 768)
        labels: (n_examples,)
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations_list = []
    labels_list = []

    for example in tqdm(examples, desc=f"Layer {layer} - Extracting"):
        text = example['text']
        target_word = example['target_word']
        label = example['label']

        # Tokenize
        tokens = model.to_tokens(text)

        # Find target position
        try:
            target_pos = find_target_token_position(
                tokens, model.tokenizer, text, target_word
            )
        except ValueError as e:
            logger.warning(f"Skipping example: {e}")
            continue

        # Run model and extract activations
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name]
            )

            # Extract activation at target position
            # cache[hook_name] is (1, seq_len, d_model)
            activation = cache[hook_name][0, target_pos, :].cpu().numpy()

            activations_list.append(activation)
            labels_list.append(label)

    activations = np.array(activations_list)
    labels = np.array(labels_list)

    logger.info(f"  Extracted {len(activations)} activations of shape {activations.shape}")

    return activations, labels


def log_diagnostics(
    activations: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    logger: logging.Logger = None
):
    """
    Log diagnostic information about activations and labels.

    Args:
        activations: (n_examples, d_model) activation matrix
        labels: (n_examples,) label array
        task_name: Name of the task for logging
        logger: Logger instance
    """
    if not logger:
        return

    logger.info(f"  [DIAGNOSTICS] {task_name}")
    logger.info(f"    Activation shape: {activations.shape}")
    logger.info(f"    Label shape: {labels.shape}")

    # Check activation statistics
    logger.info(f"    Activation mean: {activations.mean():.6f}")
    logger.info(f"    Activation std: {activations.std():.6f}")
    logger.info(f"    Activation min: {activations.min():.6f}")
    logger.info(f"    Activation max: {activations.max():.6f}")

    # Check for constant/near-constant activations
    activation_variance = activations.var(axis=0)
    zero_variance_dims = (activation_variance < 1e-10).sum()
    logger.info(f"    Dimensions with zero variance: {zero_variance_dims}/{activations.shape[1]}")

    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"    Label distribution: {dict(zip(unique_labels, counts))}")

    # Check if activations differ between classes
    if len(unique_labels) == 2:  # Binary classification
        class0_acts = activations[labels == unique_labels[0]]
        class1_acts = activations[labels == unique_labels[1]]

        mean_diff = np.abs(class0_acts.mean(axis=0) - class1_acts.mean(axis=0)).mean()
        logger.info(f"    Mean activation difference between classes: {mean_diff:.6f}")

        # Check if classes are separable at all
        total_var = activations.var()
        between_class_var = ((class0_acts.mean() - class1_acts.mean()) ** 2) / 2
        logger.info(f"    Total variance: {total_var:.6f}")
        logger.info(f"    Between-class variance: {between_class_var:.6f}")
        logger.info(f"    Separability ratio: {between_class_var / (total_var + 1e-10):.6f}")


def apply_pca_and_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_components: int = 10,
    n_runs: int = 3,
    logger: logging.Logger = None
) -> Dict:
    """
    Apply PCA and train probes to measure classification performance.

    Args:
        activations: (n_examples, 768) activation matrix
        labels: (n_examples,) label array
        n_components: Number of PCA components (default: 10)
        n_runs: Number of probe training runs (default: 3)
        logger: Logger instance

    Returns:
        Dictionary with:
        - explained_variance_ratio: per-component explained variance
        - cumulative_variance: cumulative explained variance
        - mutual_information: list of MI scores (one per run)
        - accuracy: list of accuracy scores (one per run)
        - f1_score: list of F1 scores (one per run)
    """
    # Standardize activations (mean=0, std=1 per feature)
    scaler = StandardScaler()
    standardized_activations = scaler.fit_transform(activations)

    # Fit PCA on standardized activations
    pca = PCA(n_components=n_components)
    reduced_activations = pca.fit_transform(standardized_activations)

    # Log explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    if logger:
        logger.info(f"  PCA explained variance (top {n_components} components):")
        logger.info(f"    Per-component: {explained_var}")
        logger.info(f"    Cumulative: {cumulative_var[-1]:.4f} ({cumulative_var[-1]*100:.1f}%)")

    # Train probes multiple times
    mi_scores = []
    accuracy_scores = []
    f1_scores = []

    for run in range(n_runs):
        # Train logistic regression probe with increased iterations
        probe = LogisticRegression(max_iter=2000, random_state=42 + run)
        probe.fit(reduced_activations, labels)

        # Get predictions
        predictions = probe.predict(reduced_activations)

        # Calculate metrics
        mi = mutual_info_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')

        mi_scores.append(mi)
        accuracy_scores.append(acc)
        f1_scores.append(f1)

    if logger:
        logger.info(f"  PCA Probe performance ({n_runs} runs):")
        logger.info(f"    Mutual Information: {np.mean(mi_scores):.4f} ± {np.std(mi_scores):.4f}")
        logger.info(f"    Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
        logger.info(f"    F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    return {
        'explained_variance_ratio': explained_var,
        'cumulative_variance': cumulative_var[-1],
        'mutual_information': mi_scores,
        'accuracy': accuracy_scores,
        'f1_score': f1_scores
    }


def apply_random_and_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_features: int = 384,
    n_subsets: int = 5,
    logger: logging.Logger = None
) -> Dict:
    """
    Sample random feature subsets and train probes (baseline comparison).

    Instead of PCA, randomly sample n_features dimensions from the 768-dim activations.
    Train one probe per random subset (5 subsets total for comparison).

    Args:
        activations: (n_examples, 768) activation matrix
        labels: (n_examples,) label array
        n_features: Number of random features to sample (default: 384 = width/2)
        n_subsets: Number of random subsets to try (default: 5)
        logger: Logger instance

    Returns:
        Dictionary with:
        - mutual_information: list of MI scores (one per subset)
        - accuracy: list of accuracy scores (one per subset)
        - f1_score: list of F1 scores (one per subset)
    """
    # Standardize activations first (mean=0, std=1 per feature)
    scaler = StandardScaler()
    standardized_activations = scaler.fit_transform(activations)

    d_model = standardized_activations.shape[1]  # Should be 768

    mi_scores = []
    accuracy_scores = []
    f1_scores = []

    for subset_idx in range(n_subsets):
        # Randomly sample features from standardized activations
        np.random.seed(42 + subset_idx)  # Reproducible
        selected_features = np.random.choice(d_model, size=n_features, replace=False)
        random_activations = standardized_activations[:, selected_features]

        # Train logistic regression probe with more iterations
        probe = LogisticRegression(max_iter=2000, random_state=42 + subset_idx)
        probe.fit(random_activations, labels)

        # Get predictions
        predictions = probe.predict(random_activations)

        # Calculate metrics
        mi = mutual_info_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')

        mi_scores.append(mi)
        accuracy_scores.append(acc)
        f1_scores.append(f1)

    if logger:
        logger.info(f"  Random baseline ({n_subsets} subsets of {n_features} features):")
        logger.info(f"    Mutual Information: {np.mean(mi_scores):.4f} ± {np.std(mi_scores):.4f}")
        logger.info(f"    Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
        logger.info(f"    F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    return {
        'mutual_information': mi_scores,
        'accuracy': accuracy_scores,
        'f1_score': f1_scores
    }


def create_bar_plot(
    results_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    logger: logging.Logger
):
    """
    Create bar plot with confidence intervals across layers.

    Args:
        results_df: DataFrame with results for all layers
        metric_col: Column name for the metric to plot
        ylabel: Y-axis label
        title: Plot title
        output_path: Path to save the plot
        logger: Logger instance
    """
    # Get unique layers from data
    layer_list = sorted(results_df['layer'].unique())
    means = []
    cis = []

    for layer in layer_list:
        layer_df = results_df[results_df['layer'] == layer]
        values = layer_df[metric_col].values

        # Calculate mean and 95% CI
        mean = values.mean()
        ci = stats.t.interval(
            confidence=0.95,
            df=len(values)-1,
            loc=mean,
            scale=stats.sem(values)
        )
        ci_error = mean - ci[0]  # Error bar size (symmetric)

        means.append(mean)
        cis.append(ci_error)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(layer_list, means, yerr=cis, capsize=5, alpha=0.7,
                   color='steelblue', ecolor='black', linewidth=1.5)

    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(layer_list)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe PCA experiment on GPT-2 Small"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/linear_probe_pca",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=10,
        help="Number of PCA components (default: 10)"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=3,
        help="Number of probe training runs (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("="*80)
    logger.info("LINEAR PROBE PCA EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"PCA components: {args.n_components}")
    logger.info(f"Probe runs: {args.n_runs}")
    logger.info(f"Random seed: {args.seed}")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("\n" + "="*80)
    logger.info("LOADING MODEL")
    logger.info("="*80)

    model_loader = ModelLoader(
        model_name="gpt2-small",
        layers=list(range(1, 12)),  # Layers 1-11 (skip layer 0 - input embeddings)
        device=device,
        logger=logger
    )

    model = model_loader.load_model()

    # Create datasets
    logger.info("\n" + "="*80)
    logger.info("CREATING DATASETS")
    logger.info("="*80)

    # SKIPPING plurality dataset - no signal (separability ratio = 0.0)
    # plurality_data = create_plurality_dataset()
    pos_data = create_pos_dataset()

    logger.info(f"POS dataset: {len(pos_data)} examples")
    logger.info(f"  Nouns: {sum(1 for x in pos_data if x['label'] == 0)}")
    logger.info(f"  Verbs: {sum(1 for x in pos_data if x['label'] == 1)}")
    logger.info(f"  Adjectives: {sum(1 for x in pos_data if x['label'] == 2)}")
    logger.info(f"  Adverbs: {sum(1 for x in pos_data if x['label'] == 3)}")

    # Process each layer (skip layer 0 - input embeddings)
    all_results = []

    for layer in range(1, 12):
        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER {layer}")
        logger.info(f"{'='*80}")

        # SKIPPING Plurality task - no signal (separability ratio = 0.0)
        # plurality_acts, plurality_labels = extract_activations(
        #     model, plurality_data, layer, logger
        # )
        # log_diagnostics(plurality_acts, plurality_labels, "Plurality", logger)
        # plurality_pca_results = apply_pca_and_probe(...)
        # plurality_random_results = apply_random_and_probe(...)

        # Process POS task
        logger.info("\nTask: Part of Speech (4-class Classification)")
        logger.info("-" * 80)

        pos_acts, pos_labels = extract_activations(
            model, pos_data, layer, logger
        )

        # Log diagnostics for POS task
        log_diagnostics(pos_acts, pos_labels, "Part of Speech", logger)

        # Method 1: PCA (top 10 components)
        logger.info("\n  Method: PCA (top 10 components)")
        pos_pca_results = apply_pca_and_probe(
            pos_acts,
            pos_labels,
            n_components=args.n_components,
            n_runs=args.n_runs,
            logger=logger
        )

        # Add PCA results
        for run in range(args.n_runs):
            all_results.append({
                'layer': layer,
                'task': 'pos',
                'method': 'pca',
                'run': run,
                'mutual_information': pos_pca_results['mutual_information'][run],
                'accuracy': pos_pca_results['accuracy'][run],
                'f1_score': pos_pca_results['f1_score'][run]
            })

        # Method 2: Random baseline (5 subsets of 384 features)
        logger.info("\n  Method: Random baseline (5 subsets of 384 features)")
        pos_random_results = apply_random_and_probe(
            pos_acts,
            pos_labels,
            n_features=384,  # width/2
            n_subsets=5,
            logger=logger
        )

        # Add random baseline results
        for run in range(5):
            all_results.append({
                'layer': layer,
                'task': 'pos',
                'method': 'random',
                'run': run,
                'mutual_information': pos_random_results['mutual_information'][run],
                'accuracy': pos_random_results['accuracy'][run],
                'f1_score': pos_random_results['f1_score'][run]
            })

    # Create results dataframe
    results_df = pd.DataFrame(all_results)

    # Save raw results
    results_path = output_dir / "raw_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n{'='*80}")
    logger.info(f"Raw results saved to: {results_path}")

    # Generate plots
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING PLOTS")
    logger.info("="*80)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # SKIPPING Plurality plots - no signal in task
    # plurality_df = results_df[results_df['task'] == 'plurality']
    # plurality_pca_df = plurality_df[plurality_df['method'] == 'pca']
    # plurality_random_df = plurality_df[plurality_df['method'] == 'random']

    # POS plots - PCA vs Random comparison
    pos_df = results_df[results_df['task'] == 'pos']
    pos_pca_df = pos_df[pos_df['method'] == 'pca']
    pos_random_df = pos_df[pos_df['method'] == 'random']

    logger.info("\nPOS - PCA method:")
    create_bar_plot(
        pos_pca_df,
        'mutual_information',
        'Mutual Information',
        'Part of Speech (PCA): Mutual Information Across Layers',
        plots_dir / 'pos_pca_mutual_information.png',
        logger
    )

    create_bar_plot(
        pos_pca_df,
        'accuracy',
        'Accuracy',
        'Part of Speech (PCA): Classification Accuracy Across Layers',
        plots_dir / 'pos_pca_accuracy.png',
        logger
    )

    logger.info("\nPOS - Random baseline:")
    create_bar_plot(
        pos_random_df,
        'mutual_information',
        'Mutual Information',
        'Part of Speech (Random 384): Mutual Information Across Layers',
        plots_dir / 'pos_random_mutual_information.png',
        logger
    )

    create_bar_plot(
        pos_random_df,
        'accuracy',
        'Accuracy',
        'Part of Speech (Random 384): Classification Accuracy Across Layers',
        plots_dir / 'pos_random_accuracy.png',
        logger
    )

    logger.info(f"\nGenerated 4 plots in: {plots_dir} (POS task only - 2 PCA + 2 Random baseline)")

    # Summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)

    logger.info("\nPart of Speech Task - PCA (10 components):")
    for layer in range(1, 12):
        layer_df = pos_pca_df[pos_pca_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: "
            f"MI={layer_df['mutual_information'].mean():.4f} ± {layer_df['mutual_information'].std():.4f}, "
            f"Acc={layer_df['accuracy'].mean():.4f} ± {layer_df['accuracy'].std():.4f}, "
            f"F1={layer_df['f1_score'].mean():.4f} ± {layer_df['f1_score'].std():.4f}"
        )

    logger.info("\nPart of Speech Task - Random Baseline (384 features):")
    for layer in range(1, 12):
        layer_df = pos_random_df[pos_random_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: "
            f"MI={layer_df['mutual_information'].mean():.4f} ± {layer_df['mutual_information'].std():.4f}, "
            f"Acc={layer_df['accuracy'].mean():.4f} ± {layer_df['accuracy'].std():.4f}, "
            f"F1={layer_df['f1_score'].mean():.4f} ± {layer_df['f1_score'].std():.4f}"
        )

    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
