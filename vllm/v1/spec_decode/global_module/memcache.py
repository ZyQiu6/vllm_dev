import math
import pickle
from pymemcache.client import base

class MemcachedTrieStorage:
    def __init__(self, server='localhost', port=11211):
        self.client = base.Client((server, port))
        self.prefix = "trie_"
    
    def _get_key(self, path):
        """生成memcached键"""
        return f"{self.prefix}{path}"
    
    def save_node(self, node, path=""):
        """保存节点到memcached"""
        key = self._get_key(path)
        # 序列化节点数据
        node_data = {
            'reward': node.reward,
            'children_keys': list(node.children.keys())
        }
        serialized_data = pickle.dumps(node_data)
        
        # 保存到memcached，设置过期时间（例如1小时）
        self.client.set(key, serialized_data, expire=3600)
        
        # 递归保存子节点
        for child_key, child_node in node.children.items():
            child_path = f"{path}_{child_key}" if path else child_key
            self.save_node(child_node, child_path)
    
    def load_node(self, path=""):
        """从memcached加载节点"""
        key = self._get_key(path)
        serialized_data = self.client.get(key)
        
        if serialized_data is None:
            return None
            
        node_data = pickle.loads(serialized_data)
        node = TrieNode()
        node.reward = node_data['reward']
        
        # 递归加载子节点
        for child_key in node_data['children_keys']:
            child_path = f"{path}_{child_key}" if path else child_key
            child_node = self.load_node(child_path)
            if child_node:
                node.children[child_key] = child_node
        
        return node
    
    def save_tree(self, tree, tree_name="main"):
        tree_data = {
            'subpath_index': tree.subpath_index,
            'root_path': tree_name
        }
        
        # 保存树元数据
        self.client.set(f"{self.prefix}{tree_name}_meta", 
                       pickle.dumps(tree_data))
        
        # 保存根节点及其所有子节点
        self.save_node(tree.root, tree_name)
    
    def load_tree(self, tree_name="main"):
        # 加载树元数据
        meta_data = self.client.get(f"{self.prefix}{tree_name}_meta")
        if meta_data is None:
            return None
            
        tree_data = pickle.loads(meta_data)
        
        # 创建新树并加载根节点
        tree = RewardAwareSuffixTree()
        tree.subpath_index = tree_data['subpath_index']
        tree.root = self.load_node(tree_name)
        
        return tree
    
    def delete_tree(self, tree_name="main"):
        self.client.delete(f"{self.prefix}{tree_name}_meta")