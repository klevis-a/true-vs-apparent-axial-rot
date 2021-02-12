function [angVelocity]=computeAngVelocity(time,frames)
    numFrames=size(frames,3);
    angVelocity=zeros(numFrames,3);
    angVelTensor=zeros(3,3,numFrames);
    for i=1:3
        for j=1:3
            angVelTensor(i,j,:)=velocity(time,squeeze(frames(i,j,:)));
        end
    end
    for n=1:numFrames
        angVelTensor(:,:,n)=squeeze(angVelTensor(:,:,n))*squeeze(frames(1:3,1:3,n))';
        angVelocity(n,1)=angVelTensor(3,2,n);
        angVelocity(n,2)=angVelTensor(1,3,n);
        angVelocity(n,3)=angVelTensor(2,1,n);
    end
end
