from nebula3_videoprocessing.videoprocessing.utils.singleton import Singleton
from nebula3_videoprocessing.videoprocessing.utils import consts
import json
class OntologyFactory:
    _creators = {}
    def __init__(self, metaclass=Singleton): 

        self.ontology_map = {
            'persons': consts.persons_json_path,
            'attributes': consts.attribute_json_path,
            'vg_attributes': consts.vg_attribute_json_path,
            'objects': consts.object_json_path,
            'vg_objects': consts.vg_object_json_path,
            'verbs': consts.verb_json_path,
            'vg_verbs': consts.vg_verb_json_path
        }

    def register_ontology(self, ontology_name):
        try:
            ontology_implementation = self.ontology_map[ontology_name]
        except:
                dict_keys = self.ontology_map.keys()
                raise Exception("ontology not found. please use on of these keys: {}".format(dict_keys))    

        self._creators[ontology_name] = json.load(open(ontology_implementation))


    def get_ontology(self, ontology_name):
        creator = self._creators.get(ontology_name)
        if not creator:
            try:
                self.register_ontology(ontology_name)
                creator = self._creators.get(ontology_name)
            except:
                dict_keys = self.ontology_map.keys()
                raise Exception("ontology not found. please use on of these keys: {}".format(dict_keys))

        return creator

def main():
    ontology1 = OntologyFactory().get_ontology("persons")
    # print(ontology1)

    
if __name__ == "__main__":
    # main()
    pass