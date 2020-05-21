function new_bbox = NMS_det(bbox)
    new_bbox=[];
    for i=1:size(bbox,1)
        cur=bbox(i,:);
        s1=(cur(2)-cur(1))*(cur(4)-cur(3));
        if ~ismember(cur,new_bbox)
            new_bbox=[new_bbox;cur];
        else
            continue
        end
        for j=i:size(bbox,1)
            temp=bbox(j,:);
            s2=(temp(2)-temp(1))*(temp(4)-temp(3));
            s=min(s1,s2);
            if cur(1)>temp(2) || temp(1)>cur(2) || cur(3)>temp(4) || temp(3)>cur(4)
                continue;
            end
            low_bbox=max(cur(1),temp(1));
            high_bbox=min(cur(2),temp(2));
            l_bbox=max(cur(3),temp(3));
            r_bbox=min(cur(4),temp(4));
            area_mix=(high_bbox-low_bbox)*(r_bbox-l_bbox);
            if area_mix>=s*0.6
                continue;
            else
                if cur(5)<temp(5)
                    if ~ismember(temp,new_bbox)
                        new_bbox=[new_bbox;temp];
                    end
                end
            end
        end
    end
            
    
end